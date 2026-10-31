"""Task-local wakeups and operation counts on real evidence packs, never wall time; the production-shaped
backfill (300 terminal tasks carrying the former blockers beside 10 live sharers) prints its numbers."""
import contextlib
import json
import pathlib
import time
from unittest.mock import patch

import pytest

from ouroboros import evidence_packer as packer, evidence_packs as packs, maintenance, obligations, observability, store, usage_store
from supervisor import terminal_delivery
from tests.test_evidence_packs import _handle_path, _result, _task
from tests.test_evidence_packs import root as root


def _reserve(root, task):
    with usage_store.hold(root, write=True) as txn:
        txn.write({"attempt_id": f"attempt-{task}", "kind": "attempt", "state": "reserved",
                   "task_id": task, "root_task_id": task, "model": "offline", "provider": "local",
                   "reservation_upper_bound_usd": 0.01, "pricing_known": True})


@pytest.mark.parametrize("change", ["money", "delivery", "live_row", "live_evidence"])
def test_unrelated_changes_do_not_wake_the_look_or_the_swept_task(root, monkeypatch, change):
    _task(root, "packed", blobs=1)
    assert packer.run_pass(root)["unlinked"] == 2
    packer.run_pass(root)  # record the directory revision after our own unlink
    maintenance.run_job(root, "evidence_packing")  # the job's quiet fact for this state
    assert "evidence_packing" not in maintenance.due(root)
    before = packer.task_stamps(root, "packed")
    if change == "money":
        _reserve(root, "unrelated")
    elif change == "delivery":
        assert terminal_delivery.register_pending_delivery(
            root, {"delivery_id": "owed-unrelated", "task_id": "unrelated", "chat_id": 1})
    elif change == "live_row":
        _result(root, "unrelated", "running")
    else:
        _task(root, "unrelated", status="running", blobs=1)
    assert packer.task_stamps(root, "packed") == before
    assert packer.changed_tasks(root) == ([] if change != "live_evidence" else ["unrelated"])
    # None of these is a reason to spawn the child: the job is at rest and only its facts are read.
    monkeypatch.setattr(packer, "packing_pending", lambda *a, **k: pytest.fail("an unrelated change started discovery"))
    assert maintenance.run_job(root, "evidence_packing") is None
    assert "evidence_packing" not in maintenance.due(root)


def test_a_task_closing_wakes_the_look_and_its_pass(root):
    _task(root, "later", blobs=1, status="running")
    maintenance.run_job(root, "evidence_packing")
    assert "evidence_packing" not in maintenance.due(root)
    _result(root, "later", "completed")
    assert "evidence_packing" in maintenance.due(root)
    assert packer.changed_tasks(root) == ["later"]
    assert maintenance.run_job(root, "evidence_packing")["state"] == "done"
    assert packs.published_generations(root, "later") and "evidence_packing" in maintenance.due(root)  # one more look
    assert maintenance.run_job(root, "evidence_packing")["state"] == "done"
    assert "evidence_packing" not in maintenance.due(root)


@pytest.mark.parametrize("operation", ["changed_tasks", "run_pass", "job"])
def test_no_money_or_delivery_source_is_read(root, monkeypatch, operation):
    for number in range(12):
        _task(root, f"t-{number}", status="running" if number % 2 else "completed", blobs=1)
    _reserve(root, "unrelated")
    assert terminal_delivery.register_pending_delivery(root, {"delivery_id": "owed", "task_id": "t-0", "chat_id": 1})
    from ouroboros import utils
    original_read = utils.read_json_dict

    def read(path, *args, **kwargs):
        assert path.name != "terminal_deliveries.json", "a pass read the delivery outbox"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(usage_store, "hold", lambda *a, **k: pytest.fail("a pass opened the money store"))
    monkeypatch.setattr(utils, "read_json_dict", read)
    if operation == "job":
        assert maintenance.run_job(root, "evidence_packing")["state"] == "done"
    else:
        getattr(packer, operation)(root)


def _measure(root, label):
    counts = {"case": label, "leases": 0, "verifies": 0}
    connection, verify = store.connection, packs.verify_pack

    def leased(path, *args, **kwargs):
        counts["leases"] += int(pathlib.Path(path) == store.store_path(root))
        return connection(path, *args, **kwargs)

    def observed_verify(*args, **kwargs):
        counts["verifies"] += 1
        return verify(*args, **kwargs)

    with patch.object(store, "connection", leased), patch.object(packs, "verify_pack", observed_verify):
        counts["report"] = packer.run_pass(root)
    print(json.dumps(counts, sort_keys=True), flush=True)
    assert counts["report"]["faults"] == 0
    return counts


@pytest.mark.parametrize("n", [12, 24, 48])
def test_measure_backfill_then_rest(root, n):
    for number in range(n):
        _task(root, f"settled-{number}", blobs=1)
    backfill = _measure(root, f"backfill-{n}")
    assert backfill["report"]["unlinked"] == 2 * n and backfill["verifies"] == n
    after_unlink = _measure(root, f"after-unlink-{n}")
    assert after_unlink["verifies"] == 0  # no loose members left: no generation is re-read
    stable = _measure(root, f"stable-{n}")
    assert stable["leases"] == 1 and stable["verifies"] == 0  # the census read, nothing else
    _result(root, "unrelated", "running")
    unrelated = _measure(root, f"unrelated-row-{n}")
    assert unrelated["leases"] == 1 and unrelated["verifies"] == 0


def test_no_global_scan_or_foreign_stamp_inside_unlink_locks(root, monkeypatch):
    for number in range(12):
        _task(root, f"settled-{number}", blobs=1)
    active = {"writer": False, "publication": False, "task": None}
    scans = []
    open_store, name_lock = store.open_store, store.name_lock
    loose_ids, stamps, unlink = packer._loose_task_ids, packer.task_stamps, packs.unlink_loose

    @contextlib.contextmanager
    def opened(*args, **kwargs):
        with open_store(*args, **kwargs) as txn:
            before = active["writer"]
            active["writer"] |= bool(kwargs.get("write"))
            try:
                yield txn
            finally:
                active["writer"] = before

    @contextlib.contextmanager
    def locked(root, name, **kwargs):
        with name_lock(root, name, **kwargs):
            before = active["publication"]
            active["publication"] |= name == packs.PUBLICATION_LOCK_NAME
            try:
                yield
            finally:
                active["publication"] = before

    def ids(*args):
        scans.append((active["writer"], active["publication"]))
        return loose_ids(*args)

    def observed_stamps(root, task, **kwargs):
        if active["writer"] or active["publication"]:
            assert task == active["task"], "foreign task evaluated during candidate unlink"
        return stamps(root, task, **kwargs)

    def remove(root, task, *args, **kwargs):
        active["task"] = task
        try:
            return unlink(root, task, *args, **kwargs)
        finally:
            active["task"] = None

    monkeypatch.setattr(store, "open_store", opened)
    monkeypatch.setattr(store, "name_lock", locked)
    monkeypatch.setattr(packer, "_loose_task_ids", ids)
    monkeypatch.setattr(packer, "task_stamps", observed_stamps)
    monkeypatch.setattr(packs, "unlink_loose", remove)
    report = packer.run_pass(root)
    assert report["faults"] == 0 and report["unlinked"] == 24
    assert scans and not any(writer or publication for writer, publication in scans)


@pytest.mark.parametrize("rewrite", [False, True])
def test_manifest_published_after_keep_snapshot_protects_blob(root, monkeypatch, rewrite):
    refs, _ = _task(root, "packing", blobs=1)
    _task(root, "live", status="running", blobs=1)
    if rewrite:
        observability.write_call_manifest(root, task_id="live", call_id="late", manifest={})
    write = packs.write_pack
    manifests = []
    read = packs.manifest_blob_refs

    def read_refs(root, task):
        manifests.append(task)
        return read(root, task)

    def publish(*args, **kwargs):
        result = write(*args, **kwargs)
        assert "live" in manifests, "no keep-set snapshot was taken before compression"
        observability.write_call_manifest(root, task_id="live", call_id="late", manifest={"ref": refs[0]})
        return result

    with monkeypatch.context() as edits:
        edits.setattr(packs, "manifest_blob_refs", read_refs)
        edits.setattr(packs, "write_pack", publish)
        report = packer.run_pass(root)
    assert report["faults"] == 0 and report["kept"] == report["unlinked"] == 1
    assert len([task for task in manifests if task == "live"]) == 1
    assert packs.blob_bytes(root, refs[0])
    _result(root, "live")
    assert packer.run_pass(root)["faults"] == 0
    assert not pathlib.Path(refs[0]["path"]).exists()


def test_unrelated_activity_during_a_pass_does_not_stop_the_backfill(root, monkeypatch):
    # A busy install commits rows and money between candidates; only the census's own tasks matter.
    refs = [_task(root, f"settled-{number}", blobs=1)[0][0] for number in range(6)]
    write, calls = packs.write_pack, []

    def publish(*args, **kwargs):
        result = write(*args, **kwargs)
        calls.append(1)
        _result(root, "busy", "running", step=len(calls))
        _reserve(root, f"busy-{len(calls)}")
        return result

    with monkeypatch.context() as edits:
        edits.setattr(packs, "write_pack", publish)
        report = packer.run_pass(root)
    assert report["faults"] == 0 and report["packed"] == 6 and report["unlinked"] == 12
    assert not any(pathlib.Path(ref["path"]).exists() for ref in refs)


def test_a_sharer_resumed_after_the_census_reads_its_shared_blob_from_the_pack(root, monkeypatch):
    """Resuming a terminal task under its id after the census is not a reason to stop the pass: the
    shared blob is in the packed task's verified pack, where the resumed sharer's readers find it. The
    sharer's own generation, chosen at the census, is still written (verified bytes, never a loss), but
    its loose files stay: its unlink sees the moved status and waits for its own pass."""
    raw = b'{"shared": "retained"}'
    refs, _ = _task(root, "packing", blobs=1, payloads=[raw])
    other, other_handle = _task(root, "other", blobs=1, payloads=[raw], handle=b"small")
    write, resumed = packs.write_pack, []

    def resume(*args, **kwargs):
        result = write(*args, **kwargs)
        if not resumed:
            resumed.append(_result(root, "other", "running"))
        return result

    with monkeypatch.context() as edits:
        edits.setattr(packs, "write_pack", resume)
        report = packer.run_pass(root)
    assert report["faults"] == 0 and (report["packed"], report["unlinked"], report["kept"]) == (2, 2, 0)
    assert packs.blob_bytes(root, refs[0]) == raw and packs.blob_bytes(root, other[0]) == raw
    assert len(packs.published_generations(root, "other")) == 1 and _handle_path(root, "other", other_handle).exists()
    _result(root, "other")
    assert packer.run_pass(root)["faults"] == 0
    assert not _handle_path(root, "other", other_handle).exists() and packs.blob_bytes(root, other[0]) == raw


def test_publication_journal_exists_only_for_a_sweep_and_discards_crash_residue(root):
    refs, _ = _task(root, "packing", blobs=1)
    journal = packs.packs_dir(root) / packs.PUBLICATIONS_NAME
    observability.write_call_manifest(root, task_id="packing", call_id="extra", manifest={"ref": refs[0]})
    assert not journal.exists()
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_bytes(b"incomplete old pass")
    assert packer.run_pass(root)["unlinked"] == 2
    assert not journal.exists()


def _former_blocker(root, index, task):
    """One of the facts the old settlement census waited for, round-robin."""
    kind = index % 6
    if kind == 0:
        _reserve(root, task)
    elif kind == 1:
        _result(root, task, review_operations={"op": {"surface": "multi_model_review", "state": "closed"}})
    elif kind == 2:
        _result(root, task, canonical_terminal_projection_origin="terminal_transition")
    elif kind == 3:
        obligations.add(root, "custody_open", f"invocation:{task}", {"request": {"task_id": task}})
    elif kind == 4:
        obligations.add(root, "future_body_duty", f"work:{task}", {"task_id": task})
    elif index % 12 == 5:
        assert terminal_delivery.register_pending_delivery(root, {"delivery_id": f"owed-{task}", "task_id": task, "chat_id": 1})
    else:
        _result(root, task, acceptance_debt={"debt_id": f"d-{task}"})


def test_production_shaped_backfill_packs_every_terminal_task_beside_live_sharers(root):
    """The production shape at a size the default suite affords: 18 terminal tasks, each carrying a former
    blocker (every kind three times); 3 running tasks sharing a blob with three of them. One pass packs all
    18, every ref reads back byte-identical, the shared blobs stay loose. (The 300-task shape took 2–30 min
    under load against the suite's 300 s per-test bound; the property has no size: P2 proves it for one
    sharer, this for many.)"""
    terminal, live, shared = 18, 3, {}
    refs, handles = {}, {}
    for number in range(terminal):
        task = f"done-{number:03}"
        payloads = [json.dumps({"answer": task, "body": "x" * 3000}).encode(), json.dumps({"request": task, "body": "y" * 1200}).encode()]
        refs[task], handles[task] = _task(root, task, blobs=2, payloads=payloads)
        _former_blocker(root, number, task)
    for number in range(live):
        sharer, sharing = f"live-{number:02}", f"done-{number * 7:03}"
        raw = packs.blob_bytes(root, refs[sharing][0])
        shared[sharer] = refs[sharing][0]
        refs[sharer], handles[sharer] = _task(root, sharer, status="running", blobs=2, payloads=[raw, json.dumps({"own": sharer}).encode()])
    truth = {task: [packs.blob_bytes(root, ref) for ref in items] for task, items in refs.items()}
    handle_truth = {task: packs.handle_bytes(root, task, ref["path"]) for task, ref in handles.items()}
    loose_before = sum(path.stat().st_size for path in (root / "observability" / "blobs").iterdir())
    started = time.perf_counter()
    report = packer.run_pass(root)
    elapsed = time.perf_counter() - started
    packed_bytes = sum(path.stat().st_size for path in packs.packs_dir(root).glob("*.xz"))
    loose_after = sum(path.stat().st_size for path in (root / "observability" / "blobs").iterdir())
    print(json.dumps({"case": "production-shaped", "terminal": terminal, "live": live, "elapsed_sec": round(elapsed, 2),
                      "loose_gzip_bytes_before": loose_before, "loose_gzip_bytes_after": loose_after,
                      "pack_bytes_after": packed_bytes, "raw_bytes": report["raw"], **{k: report[k] for k in packer.COUNTERS}},
                     sort_keys=True), flush=True)
    assert (report["packed"], report["faults"], report["live"], report["kept"]) == (terminal, 0, live, live)
    assert report["unlinked"] == terminal * 3 - live
    for task, items in refs.items():
        assert [packs.blob_bytes(root, ref) for ref in items] == truth[task]
        assert packs.handle_bytes(root, task, handles[task]["path"]) == handle_truth[task]
    for sharer, ref in shared.items():
        assert pathlib.Path(ref["path"]).exists(), sharer  # kept loose for the live sharer
    for sharer in shared:
        assert not packs.published_generations(root, sharer)
        assert all(pathlib.Path(ref["path"]).exists() for ref in refs[sharer]) and _handle_path(root, sharer, handles[sharer]).exists()


# ---- the census after the first pass: the rows written terminal since, never the history ------------

def _observed(monkeypatch, listed, stamped):
    """Every listing of the evidence folders and every stamp the packer takes, by task."""
    loose_ids, stamp = packer._loose_task_ids, packer._stamp
    monkeypatch.setattr(packer, "_loose_task_ids", lambda *a: listed.append(1) or loose_ids(*a))
    monkeypatch.setattr(packer, "_stamp", lambda root, task, row: stamped.append(task) or stamp(root, task, row))


def test_after_its_first_pass_the_census_is_the_rows_written_terminal_since_and_the_owed_candidates(root, monkeypatch):
    """Both directions. The first pass lists the evidence folders whole and records the counter it read as
    its position; after it, an unrelated completion (no evidence) costs no listing and no stamp, a task
    that closes with evidence is stamped and packed alone, and a candidate a pass left unfinished is owed
    another attempt by id, whatever its stamp, while its row never moves again."""
    from ouroboros import maintenance_jobs

    listed, stamped = [], []
    _observed(monkeypatch, listed, stamped)
    for number in range(4):
        _task(root, f"settled-{number}", blobs=1)
    assert maintenance.run_job(root, "evidence_packing")["state"] == "done"
    row = maintenance_jobs.job(root, "evidence_packing")
    assert listed and sorted(set(stamped)) == [f"settled-{n}" for n in range(4)]
    assert row["census"] == row["terminal"] == packer.terminal_revision(root) and row["pending"] == []
    assert packer.census_since(row, packer.terminal_revision(root), packer.store_body(root)) == row["terminal"]
    assert maintenance.run_job(root, "evidence_packing")["state"] == "done"  # the look an unlink owes
    assert "evidence_packing" not in maintenance.due(root)

    listed.clear(), stamped.clear()
    _result(root, "unrelated", "completed")  # an ordinary finished turn: its row moved, it has no evidence
    assert "evidence_packing" in maintenance.due(root)
    assert maintenance.run_job(root, "evidence_packing")["state"] == "done"
    assert (listed, stamped) == ([], []), "a completion without evidence listed or stamped the history"
    assert maintenance_jobs.job(root, "evidence_packing")["census"] == packer.terminal_revision(root)
    assert "evidence_packing" not in maintenance.due(root)

    refs, _handle = _task(root, "late", blobs=1)  # a task's own row moves, with evidence
    assert maintenance.run_job(root, "evidence_packing")["state"] == "done"
    assert listed == [] and set(stamped) == {"late"}, (listed, stamped)
    assert packs.published_generations(root, "late") and not pathlib.Path(refs[0]["path"]).exists()
    maintenance.run_job(root, "evidence_packing")
    assert "evidence_packing" not in maintenance.due(root)

    listed.clear(), stamped.clear()
    refs, _handle = _task(root, "flaky", blobs=1)
    pack_task = packs.pack_task

    def fails_once(root_, task, **kwargs):
        if task == "flaky" and not fails_once.failed:
            fails_once.failed = True
            raise OSError("the pack's temporary file could not be written")
        return pack_task(root_, task, **kwargs)

    fails_once.failed = False
    monkeypatch.setattr(packs, "pack_task", fails_once)
    assert maintenance.run_job(root, "evidence_packing")["state"] == "done"
    assert maintenance_jobs.job(root, "evidence_packing")["pending"] == ["flaky"] and pathlib.Path(refs[0]["path"]).exists()
    counter = packer.terminal_revision(root)
    listed.clear(), stamped.clear()
    assert maintenance.run_job(root, "evidence_packing")["state"] == "done"  # owed: retried by id, no row moved
    assert packer.terminal_revision(root) == counter and listed == [] and set(stamped) == {"flaky"}
    assert packs.published_generations(root, "flaky") and not pathlib.Path(refs[0]["path"]).exists()
    assert maintenance_jobs.job(root, "evidence_packing")["pending"] == []


def test_a_position_another_body_moved_or_a_counter_that_went_back_owes_a_whole_census(root, monkeypatch):
    """A body that stamps no ``terminal_seq`` moves ``terminal`` alone and counts ``pending``; a store put
    back from a copy puts the counter below the position. Either reads as no position: the next pass lists
    the folders whole and finds the unstamped task. The same row with ``census`` beside it reads addressed."""
    from ouroboros import maintenance_jobs

    listed, stamped = [], []
    _observed(monkeypatch, listed, stamped)
    _task(root, "first", blobs=1)
    maintenance.run_job(root, "evidence_packing")
    maintenance.run_job(root, "evidence_packing")
    row = maintenance_jobs.job(root, "evidence_packing")
    counter, body = packer.terminal_revision(root), packer.store_body(root)
    assert packer.census_since(row, counter, body) == counter
    assert packer.census_since({**row, "terminal": counter + 3, "census": counter + 3}, counter, body) is None  # went back
    assert packer.census_since({**row, "pending": 1}, counter, body) is None  # an older body's count of unfinished work
    assert packer.census_since({**row, "terminal": counter + 1}, counter + 1, body) is None  # moved alone
    assert packer.census_since(row, counter, {"sha": "another", "at": "2026-10-23T00:00:00+00:00"}) is None  # another HEAD booted
    assert packer.owed_ids({**row, "pending": 2}) == () and packer.owed_ids({**row, "pending": ["x"]}) == ("x",)

    refs, _handle = _task(root, "under-an-older-body", blobs=1)
    with store.open_store(root, write=True) as txn:  # the older body's terminal write: counted, never stamped
        txn.conn.execute("UPDATE tasks SET terminal_seq=NULL WHERE task_id='under-an-older-body'")
    maintenance_jobs.record(root, "evidence_packing", terminal=packer.terminal_revision(root), pending=0)  # its pass
    assert "evidence_packing" not in maintenance.due(root)  # that pass's quiet fact stands until a terminal write
    _result(root, "next", "completed")  # back under this body
    listed.clear(), stamped.clear()
    assert "evidence_packing" in maintenance.due(root)
    assert maintenance.run_job(root, "evidence_packing")["state"] == "done"
    assert listed and "under-an-older-body" in stamped
    assert packs.published_generations(root, "under-an-older-body") and not pathlib.Path(refs[0]["path"]).exists()
    row = maintenance_jobs.job(root, "evidence_packing")
    assert row["census"] == row["terminal"] and row["pending"] == []


def test_a_terminal_write_under_another_body_that_ran_no_pass_owes_one_whole_census(root, monkeypatch):
    """Both directions, through the real boot door. A Restore to an older body closes a task and returns
    before that body's packer ran: its writer moves the counter without stamping ``terminal_seq`` and leaves
    this body's row as it was, so by its own facts the row still reads addressed and the counter-keyed census
    would skip the task for good. The position is recorded under the store's body fact
    (``meta.last_store_body``, rewritten by a boot only when the HEAD differs): the first pass back lists the
    folders whole once and packs the task; this body's own reboots and the completions after it stay
    addressed (no listing, the task's stamp alone)."""
    from ouroboros import maintenance_jobs, store_import

    listed, stamped = [], []
    _observed(monkeypatch, listed, stamped)
    this_body, older_body = "c8dcd17f8-this-body", "9b1a974c0-older-body"
    store_import.ensure_current_store(root, with_children=False, body_sha=this_body)  # this body's boot
    _task(root, "first", blobs=1)
    maintenance.run_job(root, "evidence_packing")
    maintenance.run_job(root, "evidence_packing")  # the look an unlink owes
    row, body = maintenance_jobs.job(root, "evidence_packing"), packer.store_body(root)
    assert body["sha"] == this_body and row["body"] == body
    assert packer.census_since(row, row["terminal"], body) == row["terminal"]
    assert "evidence_packing" not in maintenance.due(root)

    store_import.ensure_current_store(root, with_children=False, body_sha=this_body)  # the same body boots again
    assert packer.store_body(root) == body, "the fact moved without a HEAD change"
    listed.clear(), stamped.clear()
    refs, _handle = _task(root, "later", blobs=1)
    assert maintenance.run_job(root, "evidence_packing")["state"] == "done"
    assert (listed, set(stamped)) == ([], {"later"}), "a reboot of the same body cost a whole census"
    assert packs.published_generations(root, "later") and not pathlib.Path(refs[0]["path"]).exists()
    maintenance.run_job(root, "evidence_packing")
    assert "evidence_packing" not in maintenance.due(root)

    store_import.ensure_current_store(root, with_children=False, body_sha=older_body)  # Restore: the older body boots
    refs, _handle = _task(root, "under-the-older-body", blobs=1)
    with store.open_store(root, write=True) as txn:  # its writer: the counter moved, the row carries no stamp
        txn.conn.execute("UPDATE tasks SET terminal_seq=NULL WHERE task_id='under-the-older-body'")
    store_import.ensure_current_store(root, with_children=False, body_sha=this_body)  # the update back; no pass of the older body
    row, counter = maintenance_jobs.job(root, "evidence_packing"), packer.terminal_revision(root)
    assert row["census"] == row["terminal"] < counter and row["pending"] == []  # by its own facts the row reads addressed
    assert packer.census_since(row, counter, row["body"]) == row["census"]
    assert packer.census_since(row, counter, packer.store_body(root)) is None  # not under the store's body fact
    listed.clear(), stamped.clear()
    assert "evidence_packing" in maintenance.due(root)
    assert maintenance.run_job(root, "evidence_packing")["state"] == "done"
    assert listed and "under-the-older-body" in stamped, "the older body's terminal task was never censused"
    assert packs.published_generations(root, "under-the-older-body") and not pathlib.Path(refs[0]["path"]).exists()
    row = maintenance_jobs.job(root, "evidence_packing")
    assert row["body"] == packer.store_body(root) and row["census"] == row["terminal"] == counter
    maintenance.run_job(root, "evidence_packing")
    assert "evidence_packing" not in maintenance.due(root)

    listed.clear(), stamped.clear()
    refs, _handle = _task(root, "after", blobs=1)  # one whole census, not one per pass: addressed again
    assert maintenance.run_job(root, "evidence_packing")["state"] == "done"
    assert (listed, set(stamped)) == ([], {"after"}), "the census stayed whole after the fact was recorded"
    assert packs.published_generations(root, "after") and not pathlib.Path(refs[0]["path"]).exists()


def test_the_keep_set_covers_live_sharers_whose_rows_did_not_move(root, monkeypatch):
    """Both directions. A blob a running task shares with a settled one stays loose on the settled task's
    pass; when another sharer settles later, the census is that task alone, and the running task's refs
    are still kept, read from the status and review indexes, never a listing. The same read is indexed."""
    raw = b'{"shared": "with a task that is still running"}'
    live_refs, _ = _task(root, "live", status="running", blobs=1, payloads=[raw])
    _task(root, "first", blobs=1, payloads=[raw])
    assert maintenance.run_job(root, "evidence_packing")["state"] == "done"
    maintenance.run_job(root, "evidence_packing")
    assert pathlib.Path(live_refs[0]["path"]).exists() and packs.published_generations(root, "first")
    listed = []
    monkeypatch.setattr(packer, "_loose_task_ids", lambda *a: listed.append(1))
    second_refs, _ = _task(root, "second", blobs=1, payloads=[raw])
    assert maintenance.run_job(root, "evidence_packing")["state"] == "done"
    assert listed == [] and packs.published_generations(root, "second")
    assert pathlib.Path(live_refs[0]["path"]).exists(), "the shared blob left while its sharer still runs"
    assert packs.blob_bytes(root, second_refs[0]) == raw
    _result(root, "live", "completed")
    assert maintenance.run_job(root, "evidence_packing")["state"] == "done"
    assert not pathlib.Path(live_refs[0]["path"]).exists() and packs.blob_bytes(root, live_refs[0]) == raw
    rows = packer.keep_rows(root)
    assert rows == {} and listed == []
    _result(root, "reviewed", review_operations={"op": {"surface": "multi_model_review", "state": "dispatched"}})
    _result(root, "queued", "scheduled")
    assert sorted(packer.keep_rows(root)) == ["queued", "reviewed"]
    traced = []
    with store.open_store(root, write=False) as txn:
        txn.conn.set_trace_callback(traced.append)
        try:
            packer_rows = {record["task_id"] for record in txn.conn.execute(
                "SELECT task_id FROM tasks WHERE status IN ('interrupted','requested','running','scheduled') "
                "OR review_collection_pending=1")}
            plan = [str(record[3]) for record in txn.conn.execute(
                "EXPLAIN QUERY PLAN SELECT task_id FROM tasks WHERE status IN ('interrupted','requested','running','scheduled') "
                "OR review_collection_pending=1")]
        finally:
            txn.conn.set_trace_callback(None)
    assert packer_rows == {"queued", "reviewed"}
    assert not any(detail.startswith("SCAN") for detail in plan), plan


def _sidecar_listings(monkeypatch):
    """Every ``list_sidecars`` call of a pass: the whole listings and the name lists of addressed ones."""
    whole, named, real = [], [], packs.list_sidecars

    def spy(root_, *, errors=None, names=None):
        (whole if names is None else named).append(sorted(names or ()))
        return real(root_, errors=errors, names=names)

    monkeypatch.setattr(packs, "list_sidecars", spy)
    return whole, named


def test_a_pass_from_a_position_reads_its_candidates_own_sidecars_and_an_empty_one_reads_none(root, monkeypatch):
    """Both directions beside many published packs of other tasks. A pass from a position with nothing to pack
    reads no sidecar; one with a candidate finds that task's generations by their names, and its one whole
    listing is the newest-pack mark that orders the candidates (``ordered_candidates``)."""
    for number in range(6):
        _task(root, f"foreign-{number}", blobs=1)
    assert packer.run_pass(root)["faults"] == 0  # no position: a whole census, one shared listing
    assert len(packs._sidecar_names(root)) == 6
    since = packer.terminal_revision(root)
    whole, named = _sidecar_listings(monkeypatch)
    report = packer.run_pass(root, since=since)
    assert report["candidates"] == 0 and (whole, named) == ([], [])

    _task(root, "late", blobs=1)
    report = packer.run_pass(root, since=since)
    assert (report["candidates"], report["packed"], report["faults"]) == (1, 1, 0)
    assert len(whole) == 1 and all(set(names) <= {"late.members.json"} for names in named), (whole, named)
    assert [sidecar["generation"] for sidecar in packs.published_generations(root, "late")] == [1]


def test_a_candidate_beside_its_homonym_takes_the_next_free_generation_and_keeps_to_its_own(root):
    """``a.2`` names ``a``'s generation 2 and ``a.2``'s generation 1: found by spelling, ``a``'s names include
    it, and only the sidecar's own ``task_id`` tells them apart. ``a``'s later evidence takes the next free
    generation and leaves its loose file only once its own pack holds it; ``a.2``'s pack stays its own."""
    from ouroboros.artifacts import store_actor_source_bytes

    handles = {task: _task(root, task, blobs=1, handle=f"# focus {task}\n".encode())[1] for task in ("a", "a.2")}
    assert packer.run_pass(root)["faults"] == 0
    assert packs._task_sidecar_names(root, "a") == {"a.members.json", "a.2.members.json"}
    since = packer.terminal_revision(root)
    later = store_actor_source_bytes(root, "a", category="context_checkpoints", source_id="focus_source_later",
                                     data=b"# later focus\n", extension="md")
    _result(root, "a", "completed")  # written terminal again: a candidate of the next pass
    report = packer.run_pass(root, since=since)
    assert (report["candidates"], report["packed"], report["faults"]) == (1, 1, 0)
    assert [sidecar["generation"] for sidecar in packs.published_generations(root, "a")] == [1, 3]
    assert [sidecar["generation"] for sidecar in packs.published_generations(root, "a.2")] == [1]
    assert not _handle_path(root, "a", later).exists()  # its own pack holds it now
    packs._VALIDATED.clear()  # the sidecars answer, not a cache this process validated
    assert packs.handle_bytes(root, "a", later["path"]) == b"# later focus\n"
    for task, handle in handles.items():
        assert packs.handle_bytes(root, task, handle["path"]) == f"# focus {task}\n".encode()
