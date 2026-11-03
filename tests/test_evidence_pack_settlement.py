"""Terminal = packable: a task's loose evidence is packed once its row is terminal, whatever duty, money or
delivery fact it carries; a task under an open review waits like a live one; later evidence becomes
the next generation; only the generations an unlink relies on are re-read; a referenced record that is
neither loose nor packed is reported, never invented; the census reads the store once and closes it
before any directory probe; new packs put small members first."""
from __future__ import annotations

import hashlib
import json
import lzma
import pathlib
import subprocess
import sys
import threading

import pytest

from ouroboros import evidence_packer as packer, evidence_packs as packs, obligations, observability as obs, store
from ouroboros import usage_store
from supervisor import terminal_delivery
from tests.test_evidence_packs import _blob, _handle_path, _loose, _result, _task
from tests.test_evidence_packs import root as root

LIVE = ("requested", "scheduled", "running", "interrupted")


def _money(root, task, state, monkeypatch):
    """One physical attempt of TASK in STATE through the accounting's own writers."""
    from ouroboros import usage_accounting as ua
    from ouroboros.usage_journal import IMPORT_REL

    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    (root / "state").mkdir(exist_ok=True)
    if not (root / IMPORT_REL).exists():
        (root / IMPORT_REL).write_text(json.dumps({"completed": True}), encoding="utf-8")
    reservation = ua.reserve_attempt(ua.AttemptRequest(model="openai/gpt-5.2", provider="openai", reservation_usd=1.0,
                                                       drive_root=root, task_id=task, root_task_id=task, source="test"))
    if state != "reserved":
        ua.mark_dispatched(reservation)
    if state == "unresolved":
        ua.mark_unresolved(reservation, "price unknown")
    usage_store.forget(root)


def _packable(root, task):
    with store.open_store(root, write=False) as txn:
        return packs.task_packable(txn, task)


def _rows(root, task_id):
    log = root / "logs" / "supervisor.jsonl"
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    return [row for row in rows if row.get("type") == "storage_pack" and row.get("task_id") == task_id]


BLOCKERS = {
    "money_reserved": lambda root, t, mp: _money(root, t, "reserved", mp),
    "money_dispatched": lambda root, t, mp: _money(root, t, "dispatched", mp),
    "money_unresolved": lambda root, t, mp: _money(root, t, "unresolved", mp),
    "delivery_pending": lambda root, t, mp: terminal_delivery.register_pending_delivery(
        root, {"delivery_id": f"owed-{t}", "task_id": t, "chat_id": 1}),
    "review_pointer_closed": lambda root, t, mp: _result(
        root, t, review_operations={"op": {"surface": "multi_model_review", "state": "closed"}}),
    "projection_and_synthesis_owed": lambda root, t, mp: _result(
        root, t, canonical_terminal_projection_origin="terminal_transition"),
    "acceptance_debt": lambda root, t, mp: _result(root, t, acceptance_debt={"debt_id": "d1"}),
    "pending_drive": lambda root, t, mp: obligations.add(root, "pending_drives", t),
    "custody_open": lambda root, t, mp: obligations.add(root, "custody_open", "invocation:x", {"request": {"task_id": t}}),
    "undeclared_duty": lambda root, t, mp: obligations.add(root, "future_body_duty", f"work:{t}", {"task_id": t}),
    "owner_question": lambda root, t, mp: obligations.add(root, "owner_questions", t, {"task_id": t}),
    "unknowns": lambda root, t, mp: obligations.add(root, "unknowns", f"task:{t}", {"task_id": t, "reason": "result_missing"}),
    "late_review_mark": lambda root, t, mp: obligations.add(root, "late_review_marks", "m1", {"task_id": t}),
}


@pytest.mark.parametrize("blocker", sorted(BLOCKERS))
def test_a_terminal_task_packs_whatever_former_blocker_it_carries(root, monkeypatch, blocker):
    """P1: every fact the old settlement census waited for is irrelevant to packing: the task packs, its
    loose files go, and every ref and handle reads back byte-identical from the pack."""
    refs, handle = _task(root, "done", blobs=2)
    BLOCKERS[blocker](root, "done", monkeypatch)
    raws, handle_raw = [packs.blob_bytes(root, ref) for ref in refs], packs.handle_bytes(root, "done", handle["path"])
    assert _packable(root, "done")
    report = packer.run_pass(root)
    assert (report["packed"], report["unlinked"], report["faults"]) == (1, 3, 0)
    assert not any(_loose(ref) for ref in refs) and not _handle_path(root, "done", handle).exists()
    assert [packs.blob_bytes(root, ref) for ref in refs] == raws
    assert packs.handle_bytes(root, "done", handle["path"]) == handle_raw
    packer.run_pass(root)  # the look its own unlinks owe
    assert packer.changed_tasks(root) == []


@pytest.mark.parametrize("surface", ["task_acceptance", "multi_model_review"])
def test_a_terminal_task_under_an_open_review_waits_like_a_live_one_until_the_review_closes(root, surface):
    """P1b, both directions: an open review operation of a terminal task is a live reader (an agent-session
    reviewer was handed the absolute path of the retained packet), so the task is neither packed nor its
    shared blob unlinked, the packet path stays, and the task rests; the pointer closing moves its stamp,
    and the next pass packs and unlinks everything, the packet still readable through the resolver."""
    from ouroboros.artifacts import store_actor_source_bytes, task_artifact_dir_path

    raw = b'{"shared": "evidence"}'
    done, _ = _task(root, "done", blobs=1, payloads=[raw])
    refs, handle = _task(root, "reviewed", blobs=2, payloads=[raw, b'{"own": true}'])
    packet = b'{"goal": "accept the delivered answer", "evidence": "' + b"y" * 4000 + b'"}'
    ref = store_actor_source_bytes(root, "reviewed", category="context_checkpoints", source_id="review-packet",
                                   data=packet, extension="json")
    path = (task_artifact_dir_path(root, "reviewed") / ref["path"]).resolve()  # the absolute path a session reads
    pointer = {"surface": surface, "state": "dispatched", "retry_key": "task_acceptance:d1",
               "controller": {"pid": 1, "birth": "b", "session": "s"}}
    _result(root, "reviewed", acceptance_debt={"debt_id": "d1"}, review_operations={"owner-1": pointer})
    assert _packable(root, "done") and not _packable(root, "reviewed")
    report = packer.run_pass(root)
    assert (report["packed"], report["kept"], report["reading"], report["live"], report["swept"]) == (1, 1, 1, 0, 2)
    assert report["unlinked"] == 1  # the packed task's own handle; the shared blob stays for the reader
    assert path.exists() and _loose(refs[0]) and _loose(refs[1]) and _handle_path(root, "reviewed", handle).exists()
    assert not packs.published_generations(root, "reviewed") and packs.blob_bytes(root, done[0]) == raw
    packer.run_pass(root)
    assert packer.changed_tasks(root) == []  # the reviewed task rests until its review fact moves
    _result(root, "reviewed", acceptance_debt={"debt_id": "d1"}, review_operations={"owner-1": {**pointer, "state": "closed"}})
    assert packer.changed_tasks(root) == ["reviewed"] and _packable(root, "reviewed")
    report = packer.run_pass(root)
    assert (report["packed"], report["unlinked"], report["kept"], report["reading"]) == (1, 4, 0, 0)
    assert not path.exists() and not any(_loose(r) for r in refs) and not _handle_path(root, "reviewed", handle).exists()
    assert packs.handle_bytes(root, "reviewed", ref["path"]) == packet
    assert packs.blob_bytes(root, refs[0]) == raw and packs.blob_bytes(root, refs[1]) == b'{"own": true}'


@pytest.mark.parametrize("status", LIVE)
def test_a_live_task_is_never_packed_and_its_shared_blob_waits_for_its_own_pass(root, status):
    """P2, both directions: the terminal sharer packs, the live one does not, and the blob they share
    stays loose until the live row closes; then the live task's own pass packs and unlinks it."""
    raw = b'{"shared": "window"}'
    refs, _ = _task(root, "done", blobs=1, payloads=[raw])
    live, live_handle = _task(root, "live", status=status, blobs=2, payloads=[raw, b'{"own": true}'])
    assert _packable(root, "done") and not _packable(root, "live")
    report = packer.run_pass(root)
    assert (report["packed"], report["kept"], report["live"], report["swept"]) == (1, 1, 1, 2)
    assert _loose(refs[0]) and all(_loose(ref) for ref in live) and _handle_path(root, "live", live_handle).exists()
    assert not packs.published_generations(root, "live") and packs.blob_bytes(root, refs[0]) == raw
    packer.run_pass(root)
    assert packer.changed_tasks(root) == []  # the live task rests until its status moves
    _result(root, "live", "completed")
    assert packer.changed_tasks(root) == ["live"] and _packable(root, "live")
    report = packer.run_pass(root)
    assert (report["packed"], report["unlinked"], report["kept"]) == (1, 3, 0)
    assert not _loose(refs[0]) and not any(_loose(ref) for ref in live)
    assert packs.blob_bytes(root, refs[0]) == raw and packs.blob_bytes(root, live[1]) == b'{"own": true}'


def test_late_evidence_becomes_the_next_generation_and_the_unpacker_restores_both(root):
    """P3."""
    refs, handle = _task(root, "task-a", blobs=1)
    assert packer.run_pass(root)["packed"] == 1
    late = _blob(root, json.dumps({"message": {"content": "late answer"}}).encode())
    obs.write_call_manifest(root, task_id="task-a", call_id="late", manifest={"full_payload_ref": late})
    assert packer.changed_tasks(root) == ["task-a"]
    report = packer.run_pass(root)
    assert (report["packed"], report["unlinked"]) == (1, 1)
    generations = packs.published_generations(root, "task-a")
    assert [g["generation"] for g in generations] == [1, 2]
    assert [m["sha256"] for m in generations[1]["members"]] == [late["sha256"]]
    assert not _loose(late) and packs.blob_bytes(root, late) == json.dumps({"message": {"content": "late answer"}}).encode()
    assert packs.blob_bytes(root, refs[0]).startswith(b'{"message"')
    script = root / "observability" / "packs" / packs.UNPACKER_NAME
    done = subprocess.run([sys.executable, str(script), "task-a"], capture_output=True, text=True, cwd=str(root), check=True)
    assert "restored 3 file(s) from 2 pack(s)" in done.stdout
    assert _loose(refs[0]) and _loose(late) and _handle_path(root, "task-a", handle).exists()


def test_only_the_generations_an_unlink_relies_on_are_reverified(root, monkeypatch):
    """P4: a closed older generation is never re-read (a damaged one does not block generation 3);
    an older generation that still holds a loose member is re-read, and damage there blocks that unlink."""
    verified = []
    original = packs.verify_pack

    def verify(path, sidecar):
        verified.append(path.name)
        return original(path, sidecar)

    monkeypatch.setattr(packs, "verify_pack", verify)
    refs, _ = _task(root, "gen", blobs=1)
    assert packer.run_pass(root)["packed"] == 1 and verified == ["gen.xz"]
    for number in (2, 3):
        verified.clear()
        late = _blob(root, json.dumps({"late": number}).encode())
        obs.write_call_manifest(root, task_id="gen", call_id=f"late{number}", manifest={"ref": late})
        if number == 3:
            pack = packs.packs_dir(root) / "gen.xz"
            pack.write_bytes(pack.read_bytes()[:-5] + b"XXXXX")  # nothing loose relies on it
        report = packer.run_pass(root)
        assert (report["packed"], report["unlinked"], report["faults"]) == (1, 1, 0)
        assert verified == [f"gen.{number}.xz"] and not _loose(late)
    assert [g["generation"] for g in packs.published_generations(root, "gen")] == [1, 2, 3]
    # A busy retry leaves a loose member held only by generation 1: that generation IS re-read, and
    # damage there keeps the member loose.
    busy_refs, busy_handle = _task(root, "busy", blobs=1)
    kept = pathlib.Path(busy_refs[0]["path"])
    import os
    unlink = os.unlink
    with monkeypatch.context() as patch:
        patch.setattr(os, "unlink", lambda p, *a, **k: (_ for _ in ()).throw(PermissionError("busy")) if pathlib.Path(p) == kept else unlink(p, *a, **k))
        assert packer.run_pass(root)["busy"] == 1
    assert kept.exists() and not _handle_path(root, "busy", busy_handle).exists()
    verified.clear()
    pack = packs.packs_dir(root) / "busy.xz"
    pack.write_bytes(b"damaged")
    report = packer.run_pass(root)
    assert (report["faults"], report["unlinked"]) == (1, 0) and kept.exists() and verified == ["busy.xz"]


def test_missing_records_are_reported_and_the_rest_packs(root):
    """P5: one blob removed before the pass is a counted gap with its bytes, the rest packs and reads;
    a task with refs but nothing packable still writes its row."""
    refs, handle = _task(root, "gap", blobs=3)
    pathlib.Path(refs[1]["path"]).unlink()
    gone, gone_handle = _task(root, "gone", blobs=2)
    for ref in gone:
        pathlib.Path(ref["path"]).unlink()
    _handle_path(root, "gone", gone_handle).unlink()
    report = packer.run_pass(root)
    assert (report["packed"], report["missing"], report["faults"]) == (1, 3, 0)
    [row] = _rows(root, "gap")
    assert (row["missing"], row["missing_bytes"], row["generation"]) == (1, refs[1]["size"], 1)
    assert packs.blob_bytes(root, refs[0]) and packs.blob_bytes(root, refs[2])
    with pytest.raises(FileNotFoundError):
        packs.blob_bytes(root, refs[1])
    [row] = _rows(root, "gone")
    assert (row["missing"], row["missing_bytes"], row["generation"]) == (2, sum(ref["size"] for ref in gone), None)
    assert not packs.published_generations(root, "gone")
    packer.run_pass(root)
    assert packer.changed_tasks(root) == []  # a recorded gap is not re-examined until its folder moves


@pytest.mark.parametrize("change", ["row_elsewhere", "money", "delivery", "duty", "own_manifest", "own_review"])
def test_only_this_tasks_own_evidence_defers_its_unlink(root, monkeypatch, change):
    """P6: a store write elsewhere, money, a delivery or a duty between the census and the section no
    longer defer the candidate; a new manifest of THIS task during the section still does, and so does a
    review of THIS task opening then (its reader may hold a loose path): the pack is kept, nothing unlinked."""
    refs, handle = _task(root, "packing", blobs=1)
    _task(root, "other", status="running", blobs=1)
    write = packs.write_pack

    def publish(*args, **kwargs):
        result = write(*args, **kwargs)
        if change == "row_elsewhere":
            _result(root, "other", "running", step=2)
        elif change == "money":
            _money(root, "packing", "dispatched", monkeypatch)
        elif change == "delivery":
            assert terminal_delivery.register_pending_delivery(root, {"delivery_id": "owed", "task_id": "packing", "chat_id": 1})
        elif change == "duty":
            obligations.add(root, "custody_open", "invocation:late", {"request": {"task_id": "packing"}})
        elif change == "own_review":
            _result(root, "packing", review_operations={"op": {"surface": "task_acceptance", "state": "dispatched"}})
        else:
            late = _blob(root, b'{"late": true}')
            obs.write_call_manifest(root, task_id="packing", call_id="late", manifest={"ref": late})
        return result

    with monkeypatch.context() as patch:
        patch.setattr(packs, "write_pack", publish)
        report = packer.run_pass(root)
    assert report["faults"] == 0
    if change == "own_manifest":
        assert report["unlinked"] == 0 and _loose(refs[0]) and _handle_path(root, "packing", handle).exists()
        assert "packing" in packer.changed_tasks(root)
        assert packer.run_pass(root)["unlinked"] == 3  # generation 2 for the late member, then everything goes
    elif change == "own_review":
        assert report["unlinked"] == 0 and _loose(refs[0]) and _handle_path(root, "packing", handle).exists()
        assert "packing" in packer.changed_tasks(root)
        assert packer.run_pass(root)["unlinked"] == 0 and _loose(refs[0])  # kept while the review is open
        _result(root, "packing", review_operations={"op": {"surface": "task_acceptance", "state": "closed"}})
        assert packer.run_pass(root)["unlinked"] == 2  # the generation already written serves the unlink
        assert not _loose(refs[0]) and not _handle_path(root, "packing", handle).exists()
    else:
        assert report["unlinked"] == 2 and not _loose(refs[0]) and not _handle_path(root, "packing", handle).exists()
    assert packs.blob_bytes(root, refs[0]).startswith(b'{"message"')


def test_a_folder_without_a_store_row_is_neither_packed_nor_kept_but_counted(root):
    """P7."""
    raw = b'{"shared": true}'
    refs, _ = _task(root, "done", blobs=1, payloads=[raw])
    own = _blob(root, b'{"orphan": true}')
    calls = root / "observability" / "calls" / "orphan"
    calls.mkdir(parents=True)
    (calls / "a.json").write_text(json.dumps({"ref": refs[0]}), encoding="utf-8")
    (calls / "b.json").write_text(json.dumps({"ref": own}), encoding="utf-8")
    report = packer.run_pass(root)
    assert (report["packed"], report["storeless"], report["kept"], report["unlinked"]) == (1, 1, 0, 2)
    assert not _loose(refs[0]) and packs.blob_bytes(root, refs[0]) == raw  # from the terminal task's pack
    assert _loose(own) and not packs.published_generations(root, "orphan")
    packer.run_pass(root)
    assert packer.changed_tasks(root) == []
    (calls / "c.json").write_text(json.dumps({"ref": own}), encoding="utf-8")
    assert packer.changed_tasks(root) == ["orphan"]  # looked at once more, still not packed
    report = packer.run_pass(root)
    assert (report["packed"], report["storeless"], report["swept"]) == (0, 1, 1) and _loose(own)


def test_census_reads_the_store_once_for_every_loose_task(root, monkeypatch):
    for number in range(24):
        _task(root, f"t{number}", blobs=1, status="running" if number % 2 else "completed")
    # A store transaction leases this thread's persistent connection: count the leases, not the connects.
    leased, original = [], store.connection

    def connection(path, *args, **kwargs):
        if pathlib.Path(path) == store.store_path(root):
            leased.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(store, "connection", connection)
    assert len(packer.changed_tasks(root)) == 24
    assert len(leased) == 1  # one short read for every row, not one per task
    ids = packer._loose_task_ids(root)
    rows = packer.task_rows(root, ids)
    leased.clear()
    assert len(packer.changed_stamps(root, rows=rows, ids=ids)) == 24
    assert leased == []  # the child census reuses the pass's one read


@pytest.mark.parametrize("operation", ["changed_tasks", "run_pass"])
@pytest.mark.parametrize("tier", [store.TIER_ENFORCED, store.TIER_NAME])
def test_a_writer_commits_while_a_census_directory_probe_stalls(root, monkeypatch, operation, tier):
    """No store transaction spans the census's ``stat`` calls. On the enforced tier a held read lock
    (a LockFileEx range on Windows) delays the writer's COMMIT; on the name tier the reader holds the
    one name lock; either way the 1 s writer below would fail with StoreBusy. The write lands on the
    stalled task itself: the stamp the census returns predates it, so the task stays selectable."""
    from tests.test_store import publish

    publish(root, tier=tier)
    _task(root, "stalled", blobs=1)
    _task(root, "other", blobs=1)
    target = root / "observability" / "calls" / "stalled"
    entered, release, result = threading.Event(), threading.Event(), {}
    original = pathlib.Path.stat

    def stat(self, *args, **kwargs):
        if self == target and threading.current_thread().name == "census" and not entered.is_set():
            entered.set()
            assert release.wait(10)
        return original(self, *args, **kwargs)

    def census():
        result["value"] = packer.changed_stamps(root) if operation == "changed_tasks" else packer.run_pass(root)

    monkeypatch.setattr(pathlib.Path, "stat", stat)
    worker = threading.Thread(target=census, name="census")
    worker.start()
    try:
        assert entered.wait(10)
        with store.open_store(root, write=True, wait_sec=1.0) as txn:
            txn.store_task({**txn.load("stalled"), "status": "running", "updated_at": "2026-10-12T00:00:00+00:00"})
    finally:
        release.set()
        worker.join(30)
    assert not worker.is_alive()
    if operation == "changed_tasks":
        assert set(result["value"]) == {"stalled", "other"}
        for task, stamp in result["value"].items():
            packer.record_swept(root, task, stamp)
        assert packer.changed_tasks(root) == ["stalled"]  # its status moved after the census read it
    else:
        assert result["value"]["faults"] == 0 and "stalled" in packer.changed_tasks(root)


def test_new_packs_order_checkpoints_then_blobs_smallest_first_and_old_order_stays_readable(root):
    payloads = [b"L" * 9000, b"s", b"M" * 300, b"t", b"m" * 300]
    refs, handle = _task(root, "ordered", blobs=len(payloads), payloads=payloads)
    assert packer.run_pass(root)["packed"] == 1
    sidecar = packs.read_sidecar(packs.packs_dir(root) / "ordered.members.json")
    members = sorted(sidecar["members"], key=lambda m: m["offset"])
    assert members[0]["kind"] == "checkpoint" and members[0]["path"] == handle["path"]
    blobs = [(m["size"], m["sha256"]) for m in members[1:]]
    assert blobs == sorted(blobs) and blobs != [(len(p), hashlib.sha256(p).hexdigest()) for p in payloads]
    assert [packs.blob_bytes(root, ref) for ref in refs] == payloads
    assert packs.handle_bytes(root, "ordered", handle["path"]).startswith(b"# focus")
    # A pack written before the order change (blobs in manifest order, large first) still resolves.
    raws = (b"Z" * 5000, b"a")
    old, raw_total = [_blob(root, raw) for raw in raws], sum(len(raw) for raw in raws)
    dict_size = packs.dict_size_for(raw_total)
    pack = lzma.compress(b"".join(raws), format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64,
                         filters=[{"id": lzma.FILTER_LZMA2, "preset": packs.PRESET, "dict_size": dict_size}])
    (packs.packs_dir(root) / "legacy.xz").write_bytes(pack)
    (packs.packs_dir(root) / "legacy.members.json").write_text(json.dumps({
        "task_id": "legacy", "generation": 1, "format_version": 1, "codec": "xz", "preset": packs.PRESET,
        "dict_size": dict_size, "raw_total": raw_total, "packed_total": len(pack),
        "pack_sha256": hashlib.sha256(pack).hexdigest(), "created_at": "2026-10-01T00:00:00Z",
        "members": [{"sha256": ref["sha256"], "kind": "json", "size": ref["size"], "offset": offset}
                    for ref, offset in zip(old, (0, len(raws[0])))]}), encoding="utf-8")
    for ref in old:
        pathlib.Path(ref["path"]).unlink()
    (packs.packs_dir(root) / packs.INDEX_NAME).unlink()
    packs._VALIDATED.clear()
    assert [packs.blob_bytes(root, ref) for ref in old] == list(raws)
