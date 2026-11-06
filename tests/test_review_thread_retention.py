"""A finished reviewer thread: its complete engine export is retained and verified here, then the engine
retires it through one bound operation (Claudexor wire seam v2/v3).

The engine here is an in-memory MOCK, never the real Claudexor; the joint engine->host checks cited below (S12,
S14) ran separately against real engine bytes, and nothing in this file repeats them. The mock keeps the seam's
rules: export reads write nothing, one operation has one settled outcome, a settled refusal is a 409 marked
``settled``, ``operation: null`` settles nothing and ``/retire/abandon`` fences. The ``real`` tests use the host's
own producers and readers for everything else."""

from __future__ import annotations

import hashlib
import io
import json
from types import SimpleNamespace

import pytest

from ouroboros import (
    delegate_continuation,
    delegate_custody,
    delegate_custody_current,
    owner_keep,
    review_ledger,
    store,
    task_status,
)
from ouroboros import review_thread_retention as rt
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.task_store import patch_document, read_document
from tests._store_shared import seed_result
from tests.test_store import document

TASK, THREAD, SEAT = "task-rt-1", "thr-1", "seat-a"
PLACE = f"review_thread:{THREAD}"
KEEP = {"basis": "test"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _settled(code: str, status: int = 409) -> ClaudexorUnavailable:
    exc = ClaudexorUnavailable(code, code, status_code=status)
    exc.problem = {"code": code, "message": code, "context": {"settled": True}}
    return exc


class Engine:
    """One thread of the v2 seam: items by plane, digests recomputed at acceptance, outcomes bound per operation."""

    def __init__(self, *, store_id: str = "store-1", runs=("run-1",)):
        self.store_identity, self.supported, self.purged, self.absent = store_id, True, False, False
        self.files = {"sql-1": ("sql", "sql/thread.json", b'{"thread": 1}'),
                      "run-1": ("run", "runs/run-1/events.jsonl", b'{"seq": 1}\n' * 50),
                      "nat-1": ("native", "native/claude/s.jsonl.zst", b"\x28\xb5\x2f\xfd" + b"z" * 300),
                      "prj-1": ("projection", "native/claude/s.jsonl", b'{"role": "user"}\n'),
                      "lane-1": ("lane", "lanes/thr-1/notes.md", b"lane notes")}
        self.extra = [{"itemId": "lnk-1", "plane": "lane", "path": "lanes/thr-1/link", "kind": "symlink",
                       "linkTarget": "notes.md", "sizeBytes": 8, "sha256": "sha256:" + _sha(b"notes.md"),
                       "mtimeMs": 1,
                       "projectionOf": None},
                      {"itemId": "cred-1", "plane": "lane", "path": "lanes/thr-1/.credentials.json",
                       "kind": "credential_store", "sizeBytes": None, "sha256": None, "mtimeMs": None,
                       "projectionOf": None, "linkTarget": None}]
        self.runs, self.coverage, self.gaps, self.busy = list(runs), "complete", [], False
        self.unsized = set()  # file items whose manifest sizeBytes is null (the schema allows it for any file)
        self.raw, self.schema, self.last_manifest = None, 1, b""  # a verbatim manifest body; its schemaVersion
        self.cleaned = False  # whether an acceptance answer already reports the engine's cleanup done
        self.ops, self.accepted, self.calls, self.faults, self.inflight, self.corrupt = {}, None, [], set(), [], set()

    # -- the engine's own view --
    def export_id(self) -> str:
        return "exp-" + _sha(json.dumps(sorted((k, v[1], _sha(v[2])) for k, v in self.files.items())).encode())[:12]

    def digests(self):
        rows = [(i, _sha(d), len(d)) for i, (_p, _path, d) in self.files.items()] + [
            (e["itemId"], e["sha256"].removeprefix("sha256:"), e["sizeBytes"]) for e in self.extra
            if e["kind"] == "symlink"]
        lane = [(i, s, n) for i, s, n in rows if (self.files.get(i, ("lane",))[0] == "lane")]
        return rt.export_digest(rows), rt.export_digest(lane)

    def body(self, op=None):
        return {"schemaVersion": 1, "storeId": self.store_identity, "threadId": THREAD,
                "threadState": "purged" if self.accepted else "active",
                "status": ("removed" if self.cleaned else "accepted_pending_removal") if self.accepted else "not_accepted",
                "acceptance": self.accepted, "cleanup": None, "operation": self.ops.get(op)}

    def land(self):  # a delayed request reaches the engine now
        while self.inflight:
            self._decide(*self.inflight.pop(0))

    # -- routes --
    def thread_retirement_supported(self):
        return self.supported

    def thread_export(self, thread_id, *, expected_store_id):
        self.land()
        self.calls.append(("export", thread_id))
        assert expected_store_id == self.store_identity
        if self.purged:
            raise _settled("thread_purged", 410)
        if self.absent:  # this store does not hold the thread: unbound, and no word about any other store
            raise ClaudexorUnavailable("thread_not_found", "no such thread", status_code=404)
        items = [{"itemId": i, "plane": p, "path": path, "kind": "file", "mtimeMs": 1,
                  "sizeBytes": None if i in self.unsized else len(d),
                  "sha256": "sha256:" + _sha(d) if p in ("sql", "projection") else None,  # as the engine spells it
                  "projectionOf": "nat-1" if p == "projection" else None, "linkTarget": None}
                 for i, (p, path, d) in self.files.items()] + self.extra
        if self.raw is not None:
            self.last_manifest = self.raw
            return self.raw
        self.last_manifest = json.dumps({
            "schemaVersion": self.schema, "storeId": self.store_identity, "threadId": thread_id,
            "exportId": self.export_id(), "closureDigest": "sha256:" + "c" * 64, "threadState": "active",
            "headRevision": 7, "workspaceMode": "in_place", "coverage": self.coverage,
            "retirable": self.coverage == "complete", "gaps": self.gaps, "disclosures": [],
            "closure": {"turns": [], "runs": [
                {"jobId": f"job-{r}", "runId": r, "state": "succeeded", "kind": "turn", "threadId": thread_id,
                 "turnId": "t", "continueFrom": None, "delegatedFrom": None} for r in self.runs]},
            "sessions": [], "items": items}).encode()
        return self.last_manifest

    def stream_thread_export_item(self, thread_id, export_id, item_id, sink):
        self.calls.append(("item", item_id))
        _plane, _path, data = self.files[item_id]
        if export_id != self.export_id():
            raise _settled("thread_export_stale")
        served = data[:-1] if item_id in self.corrupt else data  # a short body after its headers
        sink.write(served)
        return {"size": len(served), "sha256": _sha(served), "etag": f'"sha256:{_sha(data)}"',
                "content_length": len(data)}

    def _decide(self, thread_id, body):
        op = body["operationId"]
        if op in self.ops:
            return self.ops[op]
        content, lane = self.digests()
        code = ("thread_retirement_conflict" if self.accepted else "thread_busy" if self.busy
                else "thread_export_stale" if body["exportId"] != self.export_id()
                else "thread_lane_changed" if body["laneDigest"] != lane
                else "thread_export_stale" if body["contentDigest"] != content else "")
        if code:
            self.ops[op] = {"operationId": op, "outcome": "refused", "code": code, "message": code, "settledAt": "t"}
        else:
            self.accepted = {"operationId": op, "exportId": body["exportId"], "closureDigest": body["closureDigest"],
                             "contentDigest": content, "laneDigest": lane, "acceptedAt": "t"}
            self.ops[op] = {"operationId": op, "outcome": "accepted", "code": None, "message": None, "settledAt": "t"}
            self.purged = True
        return self.ops[op]

    def retire_thread(self, thread_id, body):
        self.land()
        assert set(body) == {"operationId", "exportId", "closureDigest", "contentDigest", "laneDigest"}  # strict
        self.calls.append(("retire", body["operationId"]))
        if "lost_before" in self.faults:  # the request never reached the engine
            self.faults.discard("lost_before")
            raise ClaudexorUnavailable("daemon_unreachable", "lost", status_code=0)
        if "delayed" in self.faults:  # it reaches the engine only later; the caller saw a timeout
            self.faults.discard("delayed")
            self.inflight.append((thread_id, body))
            raise ClaudexorUnavailable("daemon_unreachable", "timeout", status_code=0)
        outcome = self._decide(thread_id, body)
        if "lost_after" in self.faults:  # decided, answer lost
            self.faults.discard("lost_after")
            raise ClaudexorUnavailable("daemon_unreachable", "lost", status_code=0)
        if outcome["outcome"] != "accepted":
            raise _settled(outcome["code"])
        return self.body(body["operationId"])

    def abandon_thread_retirement(self, thread_id, operation_id):
        self.land()
        self.calls.append(("abandon", operation_id))
        self.ops.setdefault(operation_id, {"operationId": operation_id, "outcome": "abandoned",
                                           "code": "thread_retirement_abandoned", "message": "", "settledAt": "t"})
        return self.body(operation_id)

    def thread_retirement(self, thread_id, *, operation_id, expected_store_id):
        self.calls.append(("read", operation_id))
        return self.body(operation_id)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A settled task whose plan participant named the thread, its custody, and a ledger record naming the seat."""
    root = tmp_path / "data"
    root.mkdir()
    record_id = review_ledger.new_record_id()
    review_ledger.write_record(root, {"record_id": record_id, "task_id": TASK, "state": "settled",
                                      "surface": "intention", "verdict": {"aggregate": "PASS"},
                                      "rows": [{"seat_id": SEAT, "source_refs": []}]})
    plan = {"schema_version": 3, "revisions": [], "asks": [{"ask_id": "a1", "record_id": record_id, "participants": [
        {"seat_id": SEAT, "status": "answered", "review_thread_id": THREAD}]}]}
    seed_result(root, TASK, document(TASK, "completed", plan_record=plan))
    custody = {"runs": {"run-1": SimpleNamespace(run_id="run-1", task_id=TASK, invocation_id="inv-1", settled=True,
                                                 terminal_state="succeeded", continuation_of="",
                                                 started_ts="2999-01-01T00:00:00+00:00")}, "pending": []}
    monkeypatch.setattr(delegate_custody_current, "unreadable", lambda _root: False)
    monkeypatch.setattr(delegate_custody_current, "tree_closure", lambda _root, _tree, task="", **_kw: {
        "runs": {rid: run for rid, run in custody["runs"].items() if run.task_id == task},
        "unfinished": custody["runs"], "pending": custody["pending"]})  # every run a candidate: the predicate decides
    monkeypatch.setattr(delegate_custody, "invocation_record",
                        lambda _root, inv, **_kw: {"request": {"_thread_id": THREAD}} if inv == "inv-1" else None)
    monkeypatch.setattr(task_status, "task_subtree_has_live_queue_rows", lambda _root, _task: False)
    continuable = {"value": False}
    monkeypatch.setattr(delegate_continuation, "still_continuable", lambda *_a, **_k: continuable["value"])
    engine = Engine()
    rt.retire_finished_review_threads(root, engine=engine, live=lambda _t: False, census=set())  # activation
    world = SimpleNamespace(root=root, engine=engine, custody=custody, record_id=record_id, continuable=continuable)
    world.run = lambda **kw: rt.retire_finished_review_threads(
        root, engine=kw.get("engine", engine), live=lambda _t: False, census=kw.get("census", set()))
    world.touch = lambda: patch_document(root, TASK, lambda current: {**current, "touched": current.get("touched", 0) + 1})
    world.release = lambda: (read_document(root, TASK) or {}).get(rt.RELEASES, {}).get(THREAD, {})
    world.touch()  # written terminal after activation: the next pass looks at it
    return world


def _retired_items(root, release):
    receipt = rt.read_thread_export(root, TASK, release["receipt"])
    out = {}
    for row in receipt["items"]:
        if row.get("ref"):
            sink = io.BytesIO()
            rt.stream_retained_item(root, TASK, receipt, row["item_id"], sink)
            out[row["item_id"]] = sink.getvalue()
    return receipt, out


def test_a_complete_export_is_kept_verified_then_released_and_reads_offline(world):
    report = world.run()
    release = world.release()
    assert report["accepted"] == 1 and release["state"] == "accepted" and world.engine.purged
    assert release["content_digest"] == world.engine.accepted["contentDigest"]
    assert release["lane_digest"] == world.engine.accepted["laneDigest"]
    receipt, items = _retired_items(world.root, release)
    originals = {item: data for item, (_p, _path, data) in world.engine.files.items()}
    world.engine = None  # the source is gone: only the archive answers
    assert items == originals and receipt["verified"] and receipt["coverage"] == "complete"
    kinds = {row["item_id"]: (row["kind"], row["ref"]) for row in receipt["items"]}
    assert kinds["lnk-1"] == ("symlink", None) and kinds["cred-1"] == ("credential_store", None)
    assert next(r for r in receipt["items"] if r["item_id"] == "lnk-1")["link_target"] == "notes.md"
    manifest = review_ledger.read_source(world.root, TASK, receipt["manifest"])
    assert json.loads(manifest)["exportId"] == receipt["export_id"]
    seat = review_ledger.load_record(world.root, world.record_id)["rows"][0]
    assert [ref["role"] for ref in seat["historical_source_refs"]] == ["thread_export"]
    assert review_ledger.record_sources_resolvable(world.root, review_ledger.load_record(world.root, world.record_id))
    answer = owner_keep.set_owner_keep(world.root, TASK, keep=True, source=KEEP)
    assert f"review_thread:{THREAD}" in answer["retired"]  # accepted: never promised back
    assert answer["releases"] == {PLACE: "accepted_pending_removal"}  # acceptance observed, removal not


@pytest.mark.parametrize("fork", [False, True], ids=["another_root_task", "a_delegated_fork"])
def test_another_task_or_a_fork_reads_the_released_archive_through_its_read_file_pointers(world, monkeypatch, fork):
    """Real ``read_file`` dispatch for a task that does not own the archive; the read rules stay dispatch's."""
    import pathlib

    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    assert world.run()["accepted"] == 1
    world.engine = None  # the source is gone: only the archive answers
    release = world.release()
    receipt = rt.read_thread_export(world.root, TASK, release["receipt"])
    ledger_ref = review_ledger.load_record(world.root, world.record_id)["rows"][0]["historical_source_refs"][0]["ref"]
    repo, home, work = (world.root.parent / name for name in ("repo", "home", "work"))
    drive = world.root / "state" / "headless_tasks" / "reader" / "data" if fork else world.root
    for folder in (repo, home, work, drive):
        folder.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)
    monkeypatch.setattr("ouroboros.config.DATA_DIR", world.root)
    meta, fields = {"budget_drive_root": str(world.root), "delegation_role": "root", "root_task_id": "reader"}, {}
    if fork:
        meta.update(delegation_role="subagent", parent_task_id="parent", root_task_id="parent")
        fields["task_constraint"] = TaskConstraint(mode="local_readonly_subagent", surface="external_workspace",
                                                   write_root=str(work))
    ctx = ToolContext(repo_dir=repo, drive_root=drive, task_id="reader", budget_drive_root=str(world.root),
                      workspace_root=work, workspace_mode="external", project_id="p1", task_metadata=meta, **fields)
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    lane = next(row["ref"] for row in receipt["items"] if row["item_id"] == "lane-1")
    for ref, text in ((release["receipt"], "review_thread_export"), (ledger_ref, "review_thread_export"),
                      (receipt["manifest"], receipt["export_id"]), (lane, "lane notes")):
        assert ref["task_id"] == TASK and ref["root"] == "artifact_store"  # its identity, for programmatic readers
        assert text in registry.execute("read_file", dict(ref["read"]["arguments"]))
    assert "NOT_FOUND" in registry.execute("read_file", {"root": "artifact_store", "path": lane["path"]})  # the
    # artifact_store spelling names the reader's own task folder, never the owner's


def test_the_archive_is_flushed_to_disk_before_the_release_is_requested(world, monkeypatch):
    """The manifest and receipt bytes, then the archive's folders, are synced before the retire request leaves (each
    streamed item syncs its own bytes as it is written). A later pass reusing a manifest the first one kept (its bytes
    already stored, so not written again) still syncs the folders before its own request. This shows the order of the
    sync requests at the destructive step; it is no power-loss test."""
    import os
    import pathlib
    import stat

    from ouroboros import artifacts
    from ouroboros.artifacts import task_artifact_dir_path

    events, directories = [], {}
    write, opened, synced, retire = artifacts.write_bytes_atomic, os.open, os.fsync, world.engine.retire_thread

    def written(path, data, *, fsync=False):
        events.append(("file", pathlib.Path(path).name, fsync))
        return write(path, data, fsync=fsync)

    def open_file(path, *args, **kwargs):
        fd = opened(path, *args, **kwargs)
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            directories[fd] = pathlib.Path(path)
        else:
            directories.pop(fd, None)
        return fd

    def sync_file(fd):
        if fd in directories and stat.S_ISDIR(os.fstat(fd).st_mode):
            events.append(("dir", directories[fd]))
        return synced(fd)

    monkeypatch.setattr(artifacts, "write_bytes_atomic", written)
    monkeypatch.setattr(os, "open", open_file)
    monkeypatch.setattr(os, "fsync", sync_file)
    world.engine.retire_thread = lambda thread_id, body: events.append(("retire",)) or retire(thread_id, body)
    world.engine.corrupt.add("nat-1")  # the first pass holds on one item: its manifest and receipt are kept
    assert world.run()["waiting"] == 1 and world.release()["state"] == "held"
    first = len(events)
    world.engine.corrupt.clear()
    assert world.run()["accepted"] == 1
    names = [event[1:] for event in events if event[0] == "file" and event[1].startswith(f"thread-{THREAD}-")]
    assert all(synced for _name, synced in names), names  # no export file is ever published unsynced
    manifests = [name for name, _synced in names if name.startswith(f"thread-{THREAD}-manifest-")]
    assert len(manifests) == 1 and events.index(("file", manifests[0], True)) < first  # reused, not rewritten
    reuse = events[first:events.index(("retire",))]
    assert [event for event in reuse if event[0] == "file" and "-receipt-" in event[1]]  # its own receipt, synced
    folder = task_artifact_dir_path(world.root, TASK) / "source_handles" / rt.CATEGORY
    expected = set()
    while folder != world.root:
        expected.add(folder)
        folder = folder.parent
    expected.add(world.root)  # includes the entry naming the newly created task artifact folder and its parents
    if os.name != "nt":  # directory flush is explicitly unavailable on Windows
        assert expected <= {event[1] for event in reuse if event[0] == "dir"}


@pytest.mark.parametrize("failure", ["open", "fsync"])
def test_directory_io_failure_holds_the_archive_before_binding_and_can_retry(world, monkeypatch, failure):
    import errno
    import os
    import pathlib
    import stat

    from ouroboros.artifacts import task_artifact_dir_path

    if os.name == "nt":
        pytest.skip("directory flush is not available on Windows")
    target = task_artifact_dir_path(world.root, TASK) / "source_handles" / rt.CATEGORY
    opened, synced, failed = os.open, os.fsync, []
    directory_fds = set()

    def open_file(path, *args, **kwargs):
        is_target = pathlib.Path(path) == target
        if is_target and failure == "open":
            failed.append("open")
            raise OSError(errno.EIO, "archive directory open failed")
        fd = opened(path, *args, **kwargs)
        directory_fds.discard(fd)
        if is_target:
            directory_fds.add(fd)
        return fd

    def sync_file(fd):
        if fd in directory_fds and stat.S_ISDIR(os.fstat(fd).st_mode) and failure == "fsync":
            failed.append("fsync")
            raise OSError(errno.EIO, "archive directory sync failed")
        return synced(fd)

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", open_file)
        patch.setattr(os, "fsync", sync_file)
        report = world.run()
    assert failed == [failure]
    assert report["holds"] == {"archive_unavailable": 1} and report["waiting"] == 1, report
    assert not world.release() and not world.engine.purged
    assert not any(kind == "retire" for kind, _ in world.engine.calls)
    assert world.run()["accepted"] == 1  # same work, now that the real directory flush can finish


def test_keep_before_binding_holds_without_touching_the_engine(world):
    assert owner_keep.set_owner_keep(world.root, TASK, keep=True, source=KEEP)["status"] == "kept"
    report = world.run()
    assert report["holds"] == {"owner_keep": 1} and world.engine.calls == [] and not world.release()


@pytest.mark.parametrize("hold", ["pending", "continuable", "census"])
def test_an_unsettled_tree_or_a_continuable_line_waits_without_export(world, hold):
    if hold == "pending":
        world.custody["pending"] = [{"invocation_id": "inv-2"}]
    elif hold == "continuable":
        world.continuable["value"] = True
    report = world.run(census=None if hold == "census" else set())
    assert report["waiting"] == 1 and world.engine.calls == [] and not world.release()
    world.custody["pending"], world.continuable["value"] = [], False
    assert world.run()["accepted"] == 1  # asked again on the next pass, with no new terminal write


def test_threads_older_than_activation_are_never_candidates(world):
    world.custody["runs"]["run-1"].started_ts = "2000-01-01T00:00:00+00:00"
    report = world.run()
    assert report["holds"] == {"predates_activation": 1} and world.engine.calls == []


def test_a_partial_export_keeps_only_its_receipt_and_is_not_asked_again_unchanged(world):
    world.engine.coverage, world.engine.gaps = "partial", [{"code": "native_format_unqualified",
                                                            "subject": "claude", "message": "unqualified"}]
    world.run()
    release = world.release()
    assert release["state"] == "held" and release["gaps"] == ["native_format_unqualified"]
    assert not any(kind in ("item", "retire") for kind, _ in world.engine.calls)
    receipt = rt.read_thread_export(world.root, TASK, release["receipt"])
    assert not receipt["verified"] and all(row["ref"] is None for row in receipt["items"])
    world.engine.calls.clear()
    world.touch()
    assert world.run()["holds"] == {"held_unchanged": 1} and world.engine.calls == [("export", THREAD)]


def test_bad_bytes_bind_nothing_and_a_retry_fetches_only_the_unverified(world):
    world.engine.corrupt.add("nat-1")
    report = world.run()
    assert report["waiting"] == 1 and world.release()["state"] == "held"
    assert not any(kind == "retire" for kind, _ in world.engine.calls)
    world.engine.corrupt.clear()
    world.engine.calls.clear()
    assert world.run()["accepted"] == 1
    fetched = [item for kind, item in world.engine.calls if kind == "item"]
    assert "sql-1" not in fetched and "run-1" not in fetched and "nat-1" in fetched  # verified bytes stay


def test_a_lost_post_is_the_same_operation_on_the_next_pass(world):
    world.engine.faults.add("lost_before")
    assert world.run()["unknown"] == 1 and world.release()["state"] == "bound"
    operation = world.release()["operation_id"]
    answer = owner_keep.set_owner_keep(world.root, TASK, keep=False, source=KEEP)
    assert answer["status"] == "released" and answer["releases"] == {PLACE: "release_unconfirmed"}
    assert PLACE not in answer["unknown"] + answer["retained"] + answer["retired"]  # neither kept nor gone
    assert world.run()["accepted"] == 1
    assert [op for kind, op in world.engine.calls if kind == "retire"] == [operation, operation]


def test_a_delayed_acceptance_is_observed_not_sent_again(world):
    world.engine.faults.add("lost_after")
    world.run()
    operation = world.release()["operation_id"]
    world.engine.calls.clear()
    assert world.run()["accepted"] == 1 and world.release()["state"] == "accepted"
    assert world.engine.calls == [("read", operation)]


def test_keep_after_binding_abandons_an_unknown_operation_and_the_late_request_cannot_land(world):
    world.engine.faults.add("delayed")
    world.run()
    operation = world.release()["operation_id"]
    answer = owner_keep.set_owner_keep(world.root, TASK, keep=True, source=KEEP)
    assert (answer["status"], answer["keep"], answer["releases"]) == ("release_unconfirmed", True,
                                                                      {PLACE: "release_unconfirmed"})
    world.engine.inflight_later = world.engine.inflight[:]  # the request is still on its way
    world.engine.inflight.clear()
    assert world.run()["abandoned"] == 1 and world.release()["state"] == "abandoned"
    world.engine.inflight.extend(world.engine.inflight_later)
    world.engine.land()  # it arrives after the fence: the operation's bound outcome stands
    assert world.engine.ops[operation]["outcome"] == "abandoned" and not world.engine.purged
    answer = owner_keep.set_owner_keep(world.root, TASK, keep=True, source=KEEP)
    assert f"review_thread:{THREAD}" in answer["retained"] and answer["releases"] == {PLACE: "abandoned"}


def test_keep_after_an_acceptance_whose_answer_was_lost_reports_it_gone(world):
    world.engine.faults.add("lost_after")
    world.run()
    owner_keep.set_owner_keep(world.root, TASK, keep=True, source=KEEP)
    assert world.run()["accepted"] == 1  # the read finds it accepted: abandon is never needed
    assert not any(kind == "abandon" for kind, _ in world.engine.calls)
    answer = owner_keep.set_owner_keep(world.root, TASK, keep=True, source=KEEP)
    assert PLACE in answer["retired"] and answer["releases"] == {PLACE: "accepted_pending_removal"}


def test_a_restarted_pass_reconciles_the_bound_operation_from_the_document(world, monkeypatch):
    calls = []
    real = rt.drive_release
    monkeypatch.setattr(rt, "drive_release", lambda *a, **k: calls.append(1) or "bound")  # a crash before any POST
    world.run()
    bound = world.release()
    assert bound["state"] == "bound" and calls and world.engine.ops == {}
    monkeypatch.setattr(rt, "drive_release", real)
    assert world.run()["accepted"] == 1 and world.engine.accepted["operationId"] == bound["operation_id"]


def test_a_settled_refusal_spends_the_operation_until_the_export_changes(world):
    world.engine.busy = True
    world.run()
    refused = world.release()
    assert refused["state"] == "refused" and refused["code"] == "thread_busy"
    world.engine.busy = False
    world.touch()
    assert world.run()["holds"] == {"refused_unchanged": 1}  # the same inventory is not tried as it was
    world.engine.files["lane-1"] = ("lane", "lanes/thr-1/notes.md", b"lane notes, edited")
    world.touch()
    assert world.run()["accepted"] == 1 and world.release()["operation_id"] != refused["operation_id"]


def test_another_store_or_a_missing_route_holds(world):
    world.engine.supported = False
    assert world.run()["holds"] == {"engine_export_unsupported": 1} and not world.release()
    world.engine.supported, world.engine.store_identity = True, ""
    world.touch()
    report = world.run()
    held = world.release()  # no store to bind it to: a durable named hold, its exact manifest kept, nothing sent
    assert report["holds"] == {"store_mismatch": 1} and held["state"] == "held" and held["holds"] == ["store_mismatch"]
    assert not [kind for kind, _ in world.engine.calls if kind in ("item", "retire")]


def test_the_first_pass_reads_no_earlier_terminal_write(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    seed_result(root, "task-old", document("task-old", "completed"))
    engine = Engine()
    assert rt.retire_finished_review_threads(root, engine=engine, live=lambda _t: False, census=set()) == {
        "activated": True}
    assert rt.retire_finished_review_threads(root, engine=engine, live=lambda _t: False, census=set())["tasks"] == 0
    with store.open_store(root, write=False) as txn:
        assert txn.meta("job:runtime_retirement")[rt.JOB_FIELD]["waiting"] == [] and engine.calls == []


REAL_CONTINUABLE = delegate_continuation.still_continuable


def test_the_real_continuation_reader_lets_a_finished_root_go_and_holds_a_continuable_one(world, monkeypatch):
    monkeypatch.setattr(delegate_continuation, "still_continuable", REAL_CONTINUABLE)
    patch_document(world.root, TASK, lambda current: {**current, "status": "cancelled",
                                                      "cancel_origin": {"source": "", "reason": "server_shutdown"}})
    report = world.run()
    assert report["holds"] == {"continuable": 1} and world.engine.calls == [] and report["waiting"] == 1
    patch_document(world.root, TASK, lambda current: {**{k: v for k, v in current.items() if k != "cancel_origin"},
                                                      "status": "completed"})
    assert world.run()["accepted"] == 1  # the owner can no longer Continue a finished root


def test_the_real_continuation_reader_lets_an_author_stopped_root_go_with_its_offer_standing(world, monkeypatch):
    """Continue stays offered, yet a stop keeps a finished root's ordinary retention: no thread waits on the offer."""
    from ouroboros.owner_continue import continuation_offer
    from ouroboros.task_store import read_document

    monkeypatch.setattr(delegate_continuation, "still_continuable", REAL_CONTINUABLE)
    patch_document(world.root, TASK, lambda current: {
        **current, "status": "completed", "reason_code": "final_message", "outcome_axes": {"execution": {
            "status": "ok", "task_completion": {"action": "stop", "rationale": "Not ready."}}}})
    assert continuation_offer(read_document(world.root, TASK), TASK)["cause"] == "author_stop"
    assert world.run()["accepted"] == 1


def _revision(root):
    with store.open_store(root, write=False) as txn:
        return txn.revision()


def test_a_thread_the_engine_already_purged_is_saved_once_and_keep_reads_it_gone(world, monkeypatch):
    world.engine.purged = True  # purged through the ordinary route, without an acceptance
    report = world.run()
    assert report["holds"] == {"thread_purged": 1} and report["waiting"] == 0, report
    assert report["observed"] == [{"task_id": TASK, "thread_id": THREAD, "state": "purged"}]
    saved = world.release()  # the store's own word: no export, no operation, no acceptance is invented
    assert saved == {"state": "purged", "thread_id": THREAD, "store_id": "store-1", "code": "thread_purged",
                     "engine": {"thread_state": "purged"}, "at": saved["at"]}
    world.engine.calls.clear()
    again = world.run()  # its own write names the task once more: nothing is asked or saved again
    assert (world.engine.calls, again["holds"], again["observed"]) == ([], {}, []) and world.release() == saved
    before = _revision(world.root)
    assert world.run()["tasks"] == 0 and _revision(world.root) == before  # at rest
    told = json.loads(_tool_keep(world.root))
    assert (told["status"], told["keep"], told["retained"], told["retired"]) == ("already_retired", False, [], [PLACE])
    assert told["releases"] == {PLACE: "purged"} and told["unknown"] == []
    route = _route_keep(world.root, monkeypatch)
    assert route["status"] == "already_retired" and route["releases"] == {PLACE: "purged"}
    assert "owner_keep" not in read_document(world.root, TASK)


def test_a_scoped_not_found_is_no_purge_and_keep_claims_nothing_about_the_thread(world, monkeypatch):
    world.engine.absent = True  # 404: this store does not hold it, which says nothing of a purge or any other store
    report = world.run()
    assert report["holds"] == {"thread_not_found": 1} and report["waiting"] == 0, report
    saved = world.release()
    assert saved == {"state": "not_found", "thread_id": THREAD, "store_id": "store-1", "code": "thread_not_found",
                     "engine": {}, "at": saved["at"]}
    told = json.loads(_tool_keep(world.root))  # neither retained nor retired: the fact is saved, since it may remain
    assert (told["status"], told["keep"], told["retained"], told["retired"], told["unknown"]) == (
        "kept", True, [], [], ["review_threads"])
    assert told["releases"] == {PLACE: "not_found"} and "owner_keep" in read_document(world.root, TASK)
    world.engine.calls.clear()
    assert world.run()["holds"] == {"owner_keep": 1} and world.engine.calls == []  # the Keep stands: no engine read
    route = _route_keep(world.root, monkeypatch, keep=False)
    assert route["status"] == "released" and route["unknown"] == ["review_threads"]
    world.engine.calls.clear()
    again = world.run()  # the release was a terminal write: asked again, the same answer writes nothing
    assert world.engine.calls == [("export", THREAD)] and again["observed"] == [] and world.release() == saved
    world.engine.absent = False  # that store holds it after all: the ordinary path takes it up on its next write
    world.touch()
    assert world.run()["accepted"] == 1 and world.release()["state"] == "accepted"


@pytest.mark.parametrize("answer", ["purged", "absent"])
def test_an_answer_no_store_can_be_named_for_saves_nothing_and_waits(world, answer):
    setattr(world.engine, answer, True)
    world.engine.store_identity = ""  # the gateway proves no store: the answer is nobody's word about this thread
    report = world.run()
    assert report["holds"] == {"store_unknown": 1} and report["waiting"] == 1 and not world.release(), report
    told = owner_keep.set_owner_keep(world.root, TASK, keep=False, source=KEEP)
    assert told["retained"] == [PLACE] and told["releases"] == {PLACE: "unrecorded"} and told["retired"] == []
    world.engine.store_identity = "store-1"
    assert world.run()["waiting"] == 0
    assert world.release()["state"] == {"purged": "purged", "absent": "not_found"}[answer]


def test_a_store_mismatch_at_the_export_read_records_neither_gone_nor_retained(world, monkeypatch):
    def refused(_thread_id, *, expected_store_id):
        raise ClaudexorUnavailable("store_mismatch", "another store answers here", status_code=409)

    monkeypatch.setattr(world.engine, "thread_export", refused)
    report = world.run()
    assert report["holds"] == {"export_unavailable": 1} and report["waiting"] == 1 and report["observed"] == [], report
    assert not world.release()
    told = owner_keep.set_owner_keep(world.root, TASK, keep=False, source=KEEP)
    assert told["releases"] == {PLACE: "unrecorded"} and told["retired"] == []


def test_a_purge_after_a_kept_export_keeps_its_archive_pointer_and_invents_no_operation(world):
    world.engine.corrupt.add("nat-1")  # one item arrives short: the verified rest and the receipt stay
    assert world.run()["waiting"] == 1
    held = world.release()
    world.engine.purged = True
    assert world.run()["holds"] == {"thread_purged": 1}
    saved = world.release()
    assert (saved["state"], saved["receipt"]) == ("purged", held["receipt"])
    assert saved["engine"] == {"thread_state": "purged"} and "operation_id" not in saved
    assert rt.read_thread_export(world.root, TASK, saved["receipt"])["thread_id"] == THREAD  # still readable offline
    assert owner_keep.set_owner_keep(world.root, TASK, keep=False, source=KEEP)["releases"] == {PLACE: "purged"}


def test_a_purge_after_a_spent_refusal_is_no_acceptance(world):
    world.engine.busy = True
    assert world.run()["refused"] == 1
    refused = world.release()
    world.engine.busy, world.engine.purged = False, True
    world.touch()
    assert world.run()["holds"] == {"thread_purged": 1}
    saved = world.release()
    assert (saved["state"], saved["operation_id"]) == ("purged", refused["operation_id"]) and not world.engine.accepted
    answer = owner_keep.set_owner_keep(world.root, TASK, keep=False, source=KEEP)
    assert answer["releases"] == {PLACE: "purged"} and answer["retired"] == [PLACE]


def test_discovery_reads_only_each_unnamed_session_seats_own_operations(world, monkeypatch):
    """Keep and the pass share one exact task/seat discovery: a seat whose reply named a thread, a packet seat and a
    seat that never reached a start read nothing; a stopped seat's own operations are read by (operation, seat)."""
    from ouroboros import delegate_invocation

    row = {"plan_record": {"asks": [{"record_id": "rec-1", "participants": [
        {"seat_id": "answered", "status": "answered", "route": "agent_session", "operation_id": "op-a",
         "review_thread_id": "thr-a"},
        {"seat_id": "packet", "status": "failed", "route": "api_chat", "operation_id": "op-p"},
        {"seat_id": "never", "status": "not_dispatched", "route": "agent_session"},
        {"seat_id": "refused", "status": "not_dispatched", "route": "agent_session", "operation_id": "op-r"},
        {"seat_id": "stopped", "status": "failed", "route": "agent_session", "operation_id": "op-2",
         "superseded_operation_ids": ["op-1"]}]}]}}
    reads, answers = [], {("op-2", "stopped"): {"state": "started", "request": {"_thread_id": "thr-s"}},
                          ("op-1", "stopped"): {"state": "started", "request": {"_thread_id": "thr-s"}},
                          ("op-r", "refused"): {"state": "failed_definite", "request": {"prompt": "no thread"}}}
    monkeypatch.setattr(delegate_invocation, "operation_record",
                        lambda _root, op, seat: reads.append((op, seat)) or answers.get((op, seat)))
    named, complete = rt.seat_threads(world.root, row)
    assert named == {"thr-a": [("rec-1", "answered")], "thr-s": [("rec-1", "stopped")]} and complete
    assert reads == [("op-r", "refused"), ("op-2", "stopped"), ("op-1", "stopped")]
    assert rt.review_threads(row) == {"thr-a": [("rec-1", "answered")]}  # the reply's own word stays apart
    answers[("op-r", "refused")] = {"state": "legacy_gap", "run_id": "", "reason": "request_order_not_backfilled"}
    assert rt.seat_threads(world.root, row)[1] is False  # no store fact to read it from: unknown, never absent
    answers[("op-r", "refused")] = {"state": "pending", "request": None}  # its stored body cannot be read
    assert rt.seat_threads(world.root, row)[1] is False


# ---- the transport over a real HTTP client --------------------------------------------------------------


@pytest.fixture
def wire_client(monkeypatch):
    import httpx

    from ouroboros.gateways import claudexor as wire

    original, seen, clients = httpx.Client, [], []

    def build(respond):
        def factory(**kwargs):
            value = original(**kwargs, transport=httpx.MockTransport(lambda request: seen.append(request) or respond(request)))
            clients.append(value)
            return value

        monkeypatch.setattr(wire.httpx, "Client", factory)
        return wire.ClaudexorGateway(wire.DaemonEndpoint("127.0.0.1", 1, "fixture-token")), seen

    yield build
    for value in clients:
        value.close()


def test_the_transport_streams_exact_bytes_and_keeps_the_engine_outcome_authority(wire_client):
    import httpx

    data = b"x" * (3 * 1024 * 1024 + 7)

    def respond(request):
        path = request.url.raw_path.decode()
        if path == "/v2/operations":
            return httpx.Response(200, json={"operations": [{"id": "post:threads.id.retire", "method": "POST",
                                                             "path": "/v2/threads/:id/retire"}]})
        if "/items/" in path:
            return httpx.Response(200, content=data, headers={"ETag": f'"sha256:{_sha(data)}"'})
        if path.endswith("/retire"):
            return httpx.Response(409, json={"code": "thread_lane_changed", "message": "lane",
                                             "context": {"settled": True}})
        return httpx.Response(503, json={"code": "thread_retirement_unavailable", "message": "io"})

    gateway, seen = wire_client(respond)
    assert gateway.thread_retirement_supported()
    sink = io.BytesIO()
    got = gateway.stream_thread_export_item("thr 1", "exp:1", "item/1", sink)
    assert sink.getvalue() == data and got["sha256"] == _sha(data) and got["size"] == len(data)
    assert got["etag"].strip('"') == f"sha256:{_sha(data)}" and got["content_length"] == len(data)
    assert seen[-1].url.raw_path.decode() == "/v2/threads/thr%201/export/exp%3A1/items/item%2F1"
    with pytest.raises(ClaudexorUnavailable) as refused:
        gateway.retire_thread("thr-1", {"operationId": "op"})
    assert rt.problem_outcome(refused.value)["state"] == "refused"
    with pytest.raises(ClaudexorUnavailable) as unknown:
        gateway.abandon_thread_retirement("thr-1", "op")
    assert unknown.value.status_code == 503 and rt.problem_outcome(unknown.value) is None  # no outcome is claimed


# ---- real producer records -------------------------------------------------------------------------------------
# Every host writer and reader the holds consult is the real one here: task rows (``write_task_result``), the plan
# record's own writers, the review-operation pointer, the custody request/start/settle writers and a published queue
# document with its liveness stamp. The live probe and the census are the step's defaults. The ENGINE is still the
# in-memory MOCK above.

R, C, C2, S = "rt-root", "rt-child", "rt-child-2", "rt-successor"


def _publish_queue(root, pending=(), running=()):
    from ouroboros.utils import utc_now_iso
    from tests._shared import write_queue_document

    write_queue_document(root / "state" / "queue_snapshot.json", {
        "ts": utc_now_iso(), "pending": [{"id": task["id"], "task": task} for task in pending],
        "running": [{"id": task["id"], "task": task} for task in running]})


def _row(root, task_id, status, *, parent="", tree=R, **fields):
    from ouroboros.task_results import write_task_result

    write_task_result(root, task_id, status, root_task_id=tree, parent_task_id=parent,
                      delegation_role="subagent" if parent else "root", **fields)


def _own_thread(root, owner, tree, record_id, run_id="run-1", thread=THREAD):
    """The plan participant naming the thread and its run, through the producers' own writers."""
    from ouroboros import plan_record

    _record, revision, _created = plan_record.append_revision(root, owner, {"goal": "g", "plan": "p"}, body=False)
    plan_record.append_ask(root, owner, {"ask_id": "a1", "revision": revision["n"], "record_id": record_id,
                                         "participants": [{"seat_id": SEAT, "status": "pending",
                                                           "owner_marked": True, "dispatched": True}]})
    invocation = f"inv-{run_id}"
    assert delegate_custody.record_start_requested(
        root, run_id="", task_id=owner, invocation_id=invocation, idempotency_key=invocation, operation_id="op-1",
        max_seconds=0, request={"prompt": "review", "_use_thread": True, "_thread_id": thread}, project_id="",
        project_owned=False, route="claude", surface="plan_review", slot_id=SEAT, root_task_id=tree, parent_task_id="")
    assert delegate_custody.record_started(root, delegate_custody.RunCustody(
        run_id=run_id, task_id=owner, invocation_id=invocation, idempotency_key=invocation, route_id="claude",
        root_task_id=tree, ledger_root=str(root)), shape={"access": "readonly", "mode": "ask"})
    assert delegate_custody.emit(root, delegate_custody.SETTLED, {"run_id": run_id, "task_id": owner,
                                                                  "state": "succeeded"})
    delegate_custody._CUSTODY.clear()
    plan_record.update_participant(root, owner, ask_id="a1", seat_id=SEAT,
                                   facts={"status": "answered", "review_thread_id": thread})


@pytest.fixture
def real(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    _publish_queue(root)
    _row(root, R, "running")
    record_id = review_ledger.new_record_id()
    review_ledger.write_record(root, {"record_id": record_id, "task_id": R, "state": "settled", "surface": "intention",
                                      "verdict": {"aggregate": "PASS"}, "rows": [{"seat_id": SEAT, "source_refs": []}]})
    engine = Engine()
    assert rt.retire_finished_review_threads(root, engine=engine) == {"activated": True}
    world = SimpleNamespace(root=root, engine=engine, record_id=record_id)

    def run(pending=(), running=()):
        _publish_queue(root, pending, running)
        return rt.retire_finished_review_threads(root, engine=engine)  # the real live probe and queue census

    world.run = run
    world.release = lambda owner=R: ((read_document(root, owner) or {}).get(rt.RELEASES) or {}).get(THREAD, {})
    return world


def test_real_an_ended_root_with_a_scheduled_native_child_and_no_external_leaf_holds(real):
    _own_thread(real.root, R, R, real.record_id)
    _row(real.root, C, "scheduled", parent=R)  # a native child: no queue row, no delegated run of its own
    _row(real.root, R, "completed")
    report = real.run()
    assert report["waiting"] == 1 and real.engine.calls == [] and not real.release(), report
    _row(real.root, C, "running", parent=R)
    _row(real.root, C, "completed", parent=R)
    assert real.run()["accepted"] == 1 and real.release()["state"] == "accepted"


def test_real_a_paused_child_the_queue_holds_keeps_the_tree(real):
    _own_thread(real.root, R, R, real.record_id)
    _row(real.root, R, "completed")
    parked = {"id": C, "root_task_id": R, "parent_task_id": R, "_owner_wait_resume": {"refusal": ""}}
    report = real.run(pending=[parked])
    assert report["holds"] == {"tree_live": 1} and real.engine.calls == [] and not real.release(), report
    _row(real.root, C, "running", parent=R)  # resumed, then parked again without a queue row: its row still runs
    report = real.run()
    assert report["waiting"] == 1 and real.engine.calls == [], report


def test_real_an_unfinished_descendant_review_holds_the_owner(real):
    from ouroboros.review_operation import _update_operations

    _own_thread(real.root, R, R, real.record_id)
    _row(real.root, C, "running", parent=R)
    _row(real.root, C, "completed", parent=R)
    _update_operations(real.root, C, lambda rows: {**rows, "op-c": {"state": "retained", "controller": {}}},
                       create=True)  # the child's own review operation is still open
    _row(real.root, R, "completed")
    report = real.run()
    assert report["waiting"] == 1 and real.engine.calls == [] and not real.release(), report
    _update_operations(real.root, C, lambda rows: {**rows, "op-c": {**rows["op-c"], "state": "settled"}})
    assert real.run()["accepted"] == 1


def test_real_an_unsettled_descendant_review_run_holds_the_owner(real):
    _own_thread(real.root, R, R, real.record_id)
    _row(real.root, C, "running", parent=R)
    invocation = "inv-child-review"
    assert delegate_custody.record_start_requested(
        real.root, run_id="", task_id=C, invocation_id=invocation, idempotency_key=invocation, operation_id="op-c",
        max_seconds=0, request={"prompt": "child review"}, project_id="", project_owned=False, route="codex",
        surface="change", slot_id="seat-c", root_task_id=R, parent_task_id=R)
    assert delegate_custody.record_started(real.root, delegate_custody.RunCustody(
        run_id="run-child", task_id=C, invocation_id=invocation, idempotency_key=invocation, route_id="codex",
        root_task_id=R, ledger_root=str(real.root)), shape={"access": "readonly", "mode": "ask"})
    delegate_custody._CUSTODY.clear()
    _row(real.root, C, "completed", parent=R)
    _row(real.root, R, "completed")
    report = real.run()
    assert report["holds"] == {"custody_open": 1} and real.engine.calls == [] and not real.release(), report


def test_real_continued_root_identity_keeps_the_thread_with_its_creator(real):
    from ouroboros.owner_continue import claim_on_predecessor

    _own_thread(real.root, R, R, real.record_id)
    _row(real.root, R, "cancelled", cancel_origin={"source": "", "reason": "server_shutdown"})
    claim_on_predecessor(real.root, R, "nonce-1", {"successor_task_id": S})
    _row(real.root, S, "scheduled", tree=S)
    report = real.run(pending=[{"id": S, "root_task_id": S}])
    assert report["holds"] == {"continuable": 1} and real.engine.calls == [], report
    _row(real.root, S, "running", tree=S)
    _row(real.root, S, "completed", tree=S)  # the successor's own end offers no Continue
    assert real.run()["accepted"] == 1
    assert real.release(R)["state"] == "accepted" and not real.release(S)  # the creator's record, never the successor's
    assert f"review_thread:{THREAD}" not in str(owner_keep.set_owner_keep(real.root, S, keep=True, source=KEEP))


# The continued work is the successor's whole tree (Astra supplemental review F1): S ends, completed or stopped by
# the app's single-task Stop (``gateway/tasks.py`` stamps ``cancel_origin.source`` ``http_single``), so S offers no
# Continue of its own, while its tree still holds a child the queue runs or one only its durable row schedules, an
# open review operation or a pending delegated invocation. Each holds the creator's thread until it ends.
SUCCESSOR_ENDS = {"completed": ("completed", {}, "author_finished"),
                  "owner_stop": ("cancelled", {"cancel_origin": {"source": "http_single"}}, "stopped_by_owner")}
CONTINUED_WORK = {"live_child": "continued_tree_live", "scheduled_child": "continued_tree_unsettled",
                  "open_review": "continued_review_open", "pending_run": "continued_custody_open"}


@pytest.mark.parametrize("open_work", sorted(CONTINUED_WORK))
@pytest.mark.parametrize("successor_end", sorted(SUCCESSOR_ENDS))
def test_real_the_creator_thread_waits_for_the_continued_tree_to_end(real, open_work, successor_end):
    from ouroboros.owner_continue import claim_on_predecessor, continuation_offer, mark_claim_admitted
    from ouroboros.review_operation import _update_operations

    _own_thread(real.root, R, R, real.record_id)
    _row(real.root, R, "cancelled", cancel_origin={"source": "", "reason": "server_shutdown"})
    claim_on_predecessor(real.root, R, "nonce-1", {"successor_task_id": S})
    mark_claim_admitted(real.root, R, "nonce-1")
    status, end, refusal = SUCCESSOR_ENDS[successor_end]
    _row(real.root, S, "running", tree=S)
    _row(real.root, S, status, tree=S, **end)
    assert continuation_offer(read_document(real.root, S), S)["refusal"] == refusal  # nothing continues S itself
    pending = []
    if open_work in ("live_child", "scheduled_child"):
        _row(real.root, C, "scheduled", parent=S, tree=S)
        pending = [{"id": C, "root_task_id": S, "parent_task_id": S}] if open_work == "live_child" else []
    elif open_work == "open_review":
        _update_operations(real.root, S, lambda rows: {**rows, "op-s": {"state": "retained", "controller": {}}},
                           create=True)
    else:
        assert delegate_custody.record_start_requested(
            real.root, run_id="", task_id=S, invocation_id="inv-s", idempotency_key="inv-s", operation_id="op-s",
            max_seconds=0, request={"prompt": "successor review"}, project_id="", project_owned=False,
            route="codex", surface="change", slot_id="seat-s", root_task_id=S, parent_task_id="")
    report = real.run(pending=pending)
    assert report["holds"] == {CONTINUED_WORK[open_work]: 1} and report["waiting"] == 1, report
    assert real.engine.calls == [] and not real.release() and not real.engine.purged
    if open_work in ("live_child", "scheduled_child"):
        _row(real.root, C, "running", parent=S, tree=S)
        _row(real.root, C, "completed", parent=S, tree=S)
    elif open_work == "open_review":
        _update_operations(real.root, S, lambda rows: {**rows, "op-s": {**rows["op-s"], "state": "settled"}})
    else:
        assert delegate_custody.record_started(real.root, delegate_custody.RunCustody(
            run_id="run-s", task_id=S, invocation_id="inv-s", idempotency_key="inv-s", route_id="codex",
            root_task_id=S, ledger_root=str(real.root)), shape={"access": "readonly", "mode": "ask"})
        assert delegate_custody.emit(real.root, delegate_custody.SETTLED, {"run_id": "run-s", "task_id": S,
                                                                           "state": "succeeded"})
        delegate_custody._CUSTODY.clear()
    report = real.run()  # the continued work ended: the thread goes, its record still the creator's
    assert report["accepted"] == 1 and real.release(R)["state"] == "accepted" and not real.release(S), report


def test_real_a_timeout_retried_child_owner_releases_only_after_its_retry_ends(real):
    _row(real.root, C, "running", parent=R)
    _own_thread(real.root, C, R, real.record_id)
    _row(real.root, C2, "running", parent=R, supersedes_task_id=C, original_task_id=C, timeout_retry_from=C)
    _row(real.root, C, "interrupted", parent=R, superseded_by=C2, retry_task_id=C2)
    _row(real.root, R, "completed")
    report = real.run()  # the predecessor that names the thread is never written terminal itself
    assert real.engine.calls == [] and not real.release(C), report
    _row(real.root, C2, "completed", parent=R, supersedes_task_id=C, original_task_id=C, timeout_retry_from=C)
    report = real.run()  # its retry's end names it through the reciprocal retry links
    assert report["accepted"] == 1 and real.release(C)["state"] == "accepted" and not real.release(C2), report


def test_a_file_item_without_a_manifest_size_is_verified_by_its_own_streamed_digest(world):
    world.engine.unsized = {"run-1", "nat-1", "lane-1"}  # no size and no hash in the manifest: the transfer is the proof
    world.engine.corrupt.add("nat-1")  # a short body after its headers
    assert world.run()["waiting"] == 1 and world.release()["state"] == "held"
    world.engine.corrupt.clear()
    assert world.run()["accepted"] == 1
    receipt = rt.read_thread_export(world.root, TASK, world.release()["receipt"])
    rows = {row["item_id"]: row for row in receipt["items"]}
    for item in ("run-1", "nat-1", "lane-1"):
        data = world.engine.files[item][2]
        assert (rows[item]["size"], rows[item]["sha256"]) == (len(data), _sha(data))
    assert world.release()["content_digest"] == world.engine.accepted["contentDigest"]  # the engine's own re-hash agrees


def test_real_an_intact_ended_tree_releases_through_the_runtime_retirement_job_and_reads_offline(real, monkeypatch):
    from ouroboros import claudexor_daemon, evidence_packer, maintenance_jobs
    from ouroboros import maintenance_retirement as retirement

    _own_thread(real.root, R, R, real.record_id)
    _row(real.root, C, "running", parent=R)
    _row(real.root, C, "completed", parent=R)
    _row(real.root, R, "completed")
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda **_kw: real.engine)  # the step's own attach
    _publish_queue(real.root)
    retirement.record_startup(real.root, gaps=False)  # a server start: the job's inputs changed
    assert retirement.due(real.root, maintenance_jobs.job(real.root, retirement.JOB) or {}, {}, True)
    retirement.run(real.root, lambda: False)
    release = real.release()
    assert release["state"] == "accepted" and real.engine.purged
    assert maintenance_jobs.job(real.root, retirement.JOB)[rt.JOB_FIELD]["waiting"] == []
    evidence_packer.run_pass(real.root)  # the maintenance child's packing pass leaves this category loose
    originals = {item: data for item, (_p, _path, data) in real.engine.files.items()}
    real.engine = None  # the source is gone
    receipt = review_ledger.read_selected_source(
        real.root, R, real.record_id, SEAT, role="thread_export", part=release["export_id"],
        sha256=release["receipt"]["sha256"])  # the ledger's programmatic reader; the HTTP review door serves
    # only the response and failure roles, and a task reads the archive through each ref's read_file pointer
    receipt = json.loads(receipt)
    streamed = {}
    for row in receipt["items"]:
        if row["ref"]:
            sink = io.BytesIO()
            rt.stream_retained_item(real.root, R, receipt, row["item_id"], sink)
            streamed[row["item_id"]] = sink.getvalue()
    assert streamed == originals
    assert review_ledger.record_sources_resolvable(real.root, review_ledger.load_record(real.root, real.record_id))


def test_real_a_child_started_during_the_bulk_copy_stops_the_release_at_its_boundary(real):
    _own_thread(real.root, R, R, real.record_id)
    _row(real.root, R, "completed")
    stream, started = real.engine.stream_thread_export_item, []

    def streaming(*args):
        if not started:  # the tree changes while the bytes are moving
            started.append(_row(real.root, C, "scheduled", parent=R))
        return stream(*args)

    real.engine.stream_thread_export_item = streaming
    report = real.run()
    held = real.release()
    assert report["holds"] == {"tree_unsettled": 1} and report["waiting"] == 1, report
    assert held["state"] == "held" and held["holds"] == ["tree_unsettled"] and not real.engine.ops  # nothing bound
    _row(real.root, C, "running", parent=R)
    _row(real.root, C, "completed", parent=R)
    real.engine.calls.clear()
    assert real.run()["accepted"] == 1
    assert not [item for kind, item in real.engine.calls if kind == "item"]  # every verified item was reused


def test_real_the_tree_hold_decides_from_row_facts_and_asks_only_an_underived_body(real, monkeypatch):
    """Open review work is each member row's derived collection facts, so a hold over a wide tree decodes no member
    body (the independent review measured a pass's reads growing with members times candidates); a row an older
    writer left underived is still answered by its body, so its open review holds."""
    from ouroboros import task_store
    from ouroboros.review_operation import _update_operations

    children = [f"{C}-{index}" for index in range(6)]
    for child in children:
        _row(real.root, child, "running", parent=R)
        _row(real.root, child, "completed", parent=R)
    _own_thread(real.root, R, R, real.record_id)
    _row(real.root, R, "completed")
    document, reads, real_read = read_document(real.root, R), [], task_store.read_document
    monkeypatch.setattr(task_store, "read_document",
                        lambda root, task_id, **kw: reads.append(task_id) or real_read(root, task_id, **kw))

    def hold():
        _publish_queue(real.root)
        return rt.tree_hold(real.root, R, document, live=lambda _task: False, census=set())[0]

    assert hold() == "" and not set(reads) & set(children), reads
    _update_operations(real.root, children[0], lambda rows: {**rows, "op": {"state": "retained", "controller": {}}},
                       create=True)
    with store.open_store(real.root, write=True) as txn:  # an older writer's row: its selectors underived
        txn.conn.execute("UPDATE tasks SET review_collection_pending=NULL, plan_collection_pending=NULL "
                         "WHERE task_id=?", (children[0],))
    assert hold() == "review_open" and children[0] in reads, reads



# ---- a delegate capture ready during an export takes the maintenance child first -----------------------------------


class _EachThread(Engine):
    """Several threads on the mock store: an acceptance purges only the thread it names."""

    def thread_export(self, thread_id, *, expected_store_id):
        self.purged, self.accepted = False, None
        return super().thread_export(thread_id, expected_store_id=expected_store_id)


def _two_owners(real, monkeypatch, events):
    """R and its child C each own one finished thread; the engine records each export in EVENTS."""
    from ouroboros import claudexor_daemon

    _own_thread(real.root, R, R, real.record_id)
    _row(real.root, C, "running", parent=R)
    _own_thread(real.root, C, R, real.record_id, run_id="run-2", thread="thr-2")
    _row(real.root, C, "completed", parent=R)
    _row(real.root, R, "completed")
    engine = _EachThread(runs=("run-1", "run-2"))
    export = engine.thread_export
    engine.thread_export = lambda thread_id, **kw: events.append(("export", thread_id)) or export(thread_id, **kw)
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda **_kw: engine)
    return engine


def _held_run(root, task="cap-task"):
    """A delegated run of another tree, open: its capture request is what the server addresses."""
    _row(root, task, "running", tree=task)
    assert delegate_custody.record_start_requested(
        root, run_id="", task_id=task, invocation_id="inv-cap", idempotency_key="inv-cap", operation_id="op-cap",
        max_seconds=0, request={"prompt": "work"}, project_id="", project_owned=False, route="codex", surface="change",
        slot_id="seat-cap", root_task_id=task, parent_task_id="")
    assert delegate_custody.record_started(root, delegate_custody.RunCustody(
        run_id="run-cap", task_id=task, invocation_id="inv-cap", idempotency_key="inv-cap", route_id="codex",
        root_task_id=task, ledger_root=str(root)), shape={"access": "readonly", "mode": "ask"})
    delegate_custody._CUSTODY.clear()
    return "run-cap"


def _child_pass(monkeypatch):
    """The maintenance child's own pass over the two jobs this concerns, in their real order and with their own due
    rules; every other job is left out of the fixture."""
    from ouroboros import maintenance

    names = list(maintenance.JOBS)
    assert names.index("delegate_capture") < names.index("runtime_retirement")  # captures first, as the child runs
    monkeypatch.setattr(maintenance, "JOBS", {name: maintenance.JOBS[name]
                                              for name in ("delegate_capture", "runtime_retirement")})
    return maintenance.run


def test_real_a_capture_ready_during_an_export_takes_the_next_child_then_the_rest_resumes(real, monkeypatch):
    from ouroboros import delegate_capture_job, maintenance_jobs
    from ouroboros import maintenance_retirement as retirement

    events, requested = [], []
    engine, capture_run = _two_owners(real, monkeypatch, events), _held_run(real.root)
    stream = engine.stream_thread_export_item

    def state(owner, thread):
        return ((read_document(real.root, owner).get(rt.RELEASES) or {}).get(thread) or {}).get("state")

    def streaming(*args):  # the server addresses a capture while the first thread's bytes move
        if not requested:
            assert delegate_capture_job.request_or_result(
                real.root, delegate_custody_current.run(real.root, capture_run), generation="server")["status"] == "pending"
            requested.append(capture_run)
            events.append(("requested", capture_run))
        return stream(*args)

    def capture(root, entry, should_stop):  # the capture work itself; its job's selection, order and records are real
        events.append(("capture", entry.run_id))
        return delegate_capture_job._record(root, entry, {**entry.capture_job, "state": "ready", "child_generation": "c"})

    engine.stream_thread_export_item = streaming
    monkeypatch.setattr(delegate_capture_job, "_capture", capture)
    child = _child_pass(monkeypatch)
    _publish_queue(real.root)
    retirement.record_startup(real.root, gaps=False)  # a server start: the retirement job is due
    child(real.root)  # the first child
    exported = [thread for kind, thread in events if kind == "export"]
    assert len(exported) == 1 and events[1] == ("requested", capture_run), events  # one export, then the request
    first, rest = exported[0], ({THREAD, "thr-2"} - set(exported)).pop()
    owner = C if rest == "thr-2" else R
    assert state(R if owner == C else C, first) == "accepted"  # the export under way finished its own operation
    row = maintenance_jobs.job(real.root, retirement.JOB)
    assert row.get("next_due") is None and row[rt.JOB_FIELD]["waiting"] == [owner], row  # owed as waiting work
    assert state(owner, rest) is None and ("capture", capture_run) not in events  # the next child's turn
    events.clear()
    _publish_queue(real.root)
    child(real.root)  # the next child: the capture first, then exactly the remaining thread
    assert events == [("capture", capture_run), ("export", rest)], events
    assert state(owner, rest) == "accepted" and state(R if owner == C else C, first) == "accepted"
    row = maintenance_jobs.job(real.root, retirement.JOB)
    assert row[rt.JOB_FIELD]["waiting"] == [] and row.get("next_due") is None, row


def test_real_a_capture_its_job_could_not_take_up_never_holds_the_exports_back(real, monkeypatch):
    """A request already ready when the pass began (its disposition lock held: the capture job left it pending) is the
    next child's anyway: it is no reason to stop, so the step cannot yield forever."""
    from ouroboros import delegate_capture_job, maintenance_jobs
    from ouroboros import maintenance_retirement as retirement

    events = []
    _two_owners(real, monkeypatch, events)
    capture_run = _held_run(real.root)
    delegate_capture_job.request_or_result(real.root, delegate_custody_current.run(real.root, capture_run),
                                           generation="server")
    monkeypatch.setattr(delegate_capture_job, "_capture",
                        lambda root, entry, should_stop: events.append(("capture", entry.run_id)) or False)
    child = _child_pass(monkeypatch)
    _publish_queue(real.root)
    retirement.record_startup(real.root, gaps=False)
    child(real.root)
    assert events[0] == ("capture", capture_run) and sorted(event[1] for event in events[1:]) == ["thr-1", "thr-2"]
    assert delegate_capture_job.ready_requests(real.root)  # still pending: the next child tries it again
    row = maintenance_jobs.job(real.root, retirement.JOB)
    assert row[rt.JOB_FIELD]["waiting"] == [] and row.get("next_due") is None, row

# ---- names and holds found by the joint engine->host check ----------------------------------------------------
# S14 counterexample (joint check run-20261102T172727Z; Claudexor 6e8cfbd x host f3bba8a): a complete, retirable
# export listed the legal POSIX lane file below (41 bytes, no manifest hash). The host answered
# ``export_item_malformed`` in its transient report only: no release record, no retained manifest or receipt and
# no waiting entry, so the cause vanished from the task and Keep inventory while the thread stayed in the engine.

S14_PATH = "lanes/th-2a54ee87e954/fake-session-default/home/odd\\name.txt"
LEGAL_NAMES = {  # legal relative POSIX placements: kept as receipt metadata, never a filesystem destination
    "lane-s14": S14_PATH,
    "lane-dir": "lanes/thr-1/dir\\x/name.txt",
    "lane-ext": "lanes/thr-1/name.t\\xt",
    "lane-end": "lanes/thr-1/trailing\\",
    "lane-only": "lanes/thr-1/\\\\",
    "lane-text": "lanes/thr-1/sp ace/ünï\ncode.txt",
}


def test_s14_a_backslash_lane_file_of_a_complete_export_is_retired_with_its_name_kept(world):
    world.engine.files["lane-s14"] = ("lane", S14_PATH, b"x" * 41)
    report = world.run()
    assert report["accepted"] == 1 and report["holds"] == {}, report
    receipt = rt.read_thread_export(world.root, TASK, world.release()["receipt"])
    assert next(row for row in receipt["items"] if row["item_id"] == "lane-s14")["path"] == S14_PATH


def test_legal_posix_names_are_receipt_metadata_never_storage_names(world):
    from ouroboros.artifacts import task_artifact_dir_path

    for item, path in LEGAL_NAMES.items():
        world.engine.files[item] = ("lane", path, f"bytes of {path!r}".encode())
    world.engine.unsized |= {"lane-dir", "lane-end"}
    report = world.run()
    assert report["accepted"] == 1, report
    receipt = rt.read_thread_export(world.root, TASK, world.release()["receipt"])
    rows = {row["item_id"]: row for row in receipt["items"]}
    for item, path in LEGAL_NAMES.items():
        ref = rows[item]["ref"]
        assert rows[item]["path"] == path and ref["path"].startswith("source_handles/delegated_activity/")
        assert not any(char in ref["path"] for char in "\\\n "), ref["path"]
        sink = io.BytesIO()
        rt.stream_retained_item(world.root, TASK, receipt, item, sink)
        assert sink.getvalue() == world.engine.files[item][2]
    stored = task_artifact_dir_path(world.root, TASK) / "source_handles" / "delegated_activity"
    assert not [entry.name for entry in stored.iterdir() if any(char in entry.name for char in "\\\n ")]
    assert world.release()["lane_digest"] == world.engine.accepted["laneDigest"]


@pytest.mark.parametrize("fault, code", [
    ("/lanes/thr-1/absolute", "export_item_malformed"),
    ("schema", "export_schema_unsupported"),
    ("bytes", "export_manifest_unreadable"), ("closure", "export_identity_mismatch"),
])
def test_a_malformed_export_is_a_durable_named_hold_with_its_exact_manifest(world, fault, code):
    original = world.engine.files["lane-1"]
    if fault == "schema":
        world.engine.schema = 2
    elif fault == "bytes":
        world.engine.raw = b"\x00 not a manifest"
    elif fault in ("plane", "kind", "closure"):  # well-formed JSON, another type where the seam names a string or list
        manifest = json.loads(world.engine.thread_export(THREAD, expected_store_id=world.engine.store_identity))
        world.engine.calls.clear()
        if fault == "closure":
            manifest["closure"]["runs"] = 1
        else:
            next(item for item in manifest["items"] if item["itemId"] == "lane-1")[fault] = []
        world.engine.raw = json.dumps(manifest).encode()
    else:
        world.engine.files["lane-1"] = ("lane", fault, original[2])
    report = world.run()
    held = world.release()
    assert report["holds"] == {code: 1} and report["waiting"] == 0, report
    assert held["state"] == "held" and held["holds"] == [code], held
    assert not [kind for kind, _ in world.engine.calls if kind in ("item", "retire")]  # nothing moved or released
    receipt = rt.read_thread_export(world.root, TASK, held["receipt"])
    assert receipt["holds"] == [code] and not receipt["verified"]
    assert review_ledger.read_source(world.root, TASK, receipt["manifest"]) == world.engine.last_manifest
    if fault.startswith(("lanes", "/")):
        assert receipt["detail"] == {"item_id": "lane-1", "field": "path"}
    elif fault in ("plane", "kind", "closure"):
        assert receipt["detail"] == {"item_id": None if fault == "closure" else "lane-1", "field": fault}
    keep = owner_keep.set_owner_keep(world.root, TASK, keep=False, source=KEEP)
    assert PLACE in keep["retained"] and keep["releases"] == {PLACE: "held"}  # discoverable, still the engine's
    world.engine.calls.clear()
    world.touch()
    assert world.run()["holds"] == {"held_unchanged": 1} and world.engine.calls == [("export", THREAD)]
    world.engine.schema, world.engine.raw, world.engine.files["lane-1"] = 1, None, original
    world.touch()
    assert world.run()["accepted"] == 1  # a changed export is examined again


# ---- S12: a keep saved after the release was bound -------------------------------------------------------------
# Joint check S12 (run-20261102T172727Z): ``set_owner_keep`` ran after the host's last keep check and before the
# retire POST reached the engine. The answer said top-level ``kept`` with the thread under ``unknown``, then the
# engine accepted and purged. Revision 10: the keep is saved for everything still owned and as the intention to stop
# the release; the bound thread is ``release_unconfirmed``, never kept; a retire already delivered can still win.


def _tool_keep(root, keep=True, task=TASK):
    """The real ``set_task_retention`` on a held owner message, exactly as Main calls it."""
    from ouroboros.tools.task_retention import _set_task_retention
    from tests.test_owner_keep_closure import _owner_context

    ctx, _address = _owner_context(root)
    return _set_task_retention(ctx, task, keep, "the owner wants this task kept")


def _route_keep(root, monkeypatch, keep=True, task=TASK):
    """The real ``POST /api/tasks/{task_id}/owner-keep`` route, its own request stamped as the source."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.task_owner_keep import api_task_owner_keep

    monkeypatch.setattr("ouroboros.gateway.task_owner_keep.request_drive_root", lambda _request: root)
    monkeypatch.setattr("ouroboros.gateway.task_owner_keep.migrating_error", lambda _root: None)
    client = TestClient(Starlette(routes=[Route("/api/tasks/{task_id}/owner-keep", endpoint=api_task_owner_keep,
                                                methods=["POST"])]))
    response = client.post(f"/api/tasks/{task}/owner-keep", json={"keep": keep})
    assert response.status_code == 200
    return response.json()


def test_s12_a_keep_saved_after_binding_is_release_unconfirmed_and_a_delivered_retire_can_win(world):
    retire, answers = world.engine.retire_thread, []

    def keep_then_decide(thread_id, body):  # the owner's keep lands after the host's last check, before the decision
        answers.append(_tool_keep(world.root))
        return retire(thread_id, body)

    world.engine.retire_thread = keep_then_decide
    report = world.run()
    saved = json.loads(answers[0])  # the writer's own answer, never a no-effect "nothing was saved"
    assert (saved["status"], saved["keep"], saved["releases"]) == ("release_unconfirmed", True,
                                                                   {PLACE: "release_unconfirmed"})
    assert saved["unknown"] == saved["retained"] == saved["retired"] == []  # the only material: never bare kept
    assert "owner_keep" in read_document(world.root, TASK)
    assert report["accepted"] == 1 and world.engine.purged  # best effort: the retire was already delivered
    after = owner_keep.set_owner_keep(world.root, TASK, keep=True, source=KEEP)
    assert (after["status"], after["keep"], after["releases"]) == ("already_retired", True,
                                                                    {PLACE: "accepted_pending_removal"})


def test_s12_a_keep_the_engine_has_not_decided_yet_stops_the_release_on_the_next_pass(world, monkeypatch):
    world.engine.faults.add("delayed")  # the retire is still on its way
    world.run()
    answer = _route_keep(world.root, monkeypatch)
    assert (answer["status"], answer["keep"], answer["releases"]) == ("release_unconfirmed", True,
                                                                      {PLACE: "release_unconfirmed"})
    late, world.engine.inflight = world.engine.inflight[:], []
    assert world.run()["abandoned"] == 1  # the existing reconciliation: read the operation, then abandon it
    world.engine.inflight.extend(late)
    world.engine.land()  # it arrives after the fence: refused, never accepted
    assert not world.engine.purged
    final = _route_keep(world.root, monkeypatch)
    assert (final["status"], final["retained"], final["releases"]) == ("kept", [PLACE], {PLACE: "abandoned"})


@pytest.mark.parametrize("other", ["retired", "retained"])
def test_a_keep_beside_a_bound_release_saves_and_names_every_other_material(world, other):
    drive = world.root / "elsewhere" / "child-drive"
    if other == "retained":
        drive.mkdir(parents=True)
    patch_document(world.root, TASK, lambda current: {**current, "child_drive_root": str(drive)})
    world.engine.faults.add("lost_before")
    world.run()
    answer = json.loads(_tool_keep(world.root))
    assert (answer["status"], answer["keep"]) == ("release_unconfirmed", True), answer  # never already_retired
    assert answer[other] == [str(drive)] and answer["releases"] == {PLACE: "release_unconfirmed"}
    assert "owner_keep" in read_document(world.root, TASK)  # it protects what is still owned
    released = json.loads(_tool_keep(world.root, keep=False))
    assert (released["status"], released["keep"], released["releases"]) == ("released", False,
                                                                             {PLACE: "release_unconfirmed"})
    assert "owner_keep" not in read_document(world.root, TASK)
    refused = owner_keep.set_owner_keep(world.root, TASK, keep=True, source={})
    assert refused["status"] == "refused" and "owner_keep" not in read_document(world.root, TASK)


def test_a_busy_custody_gate_saves_nothing_and_says_so(world, monkeypatch):
    import functools
    import threading

    from ouroboros.task_custody import task_custody_lock

    world.engine.faults.add("lost_before")
    world.run()
    monkeypatch.setattr(owner_keep, "set_owner_keep", functools.partial(owner_keep.set_owner_keep, timeout_sec=0.2))
    held, done = threading.Event(), threading.Event()

    def retirer():
        with task_custody_lock(world.root, TASK, timeout_sec=5) as locked:
            assert locked
            held.set()
            done.wait(10)

    thread = threading.Thread(target=retirer)
    thread.start()
    held.wait(5)
    try:
        busy = _tool_keep(world.root)
    finally:
        done.set()
        thread.join(10)
    assert "TASK_RETENTION_INFLIGHT: custody_busy. Nothing was saved." in busy
    assert "owner_keep" not in read_document(world.root, TASK)
    saved = json.loads(_tool_keep(world.root))  # the same call once the gate is free: saved, release unconfirmed
    assert saved["status"] == "release_unconfirmed" and "owner_keep" in read_document(world.root, TASK)


# ---- a keep before any release is recorded (independent review R1, R2, R2b) -----------------------------------
# The keep inventory once named only threads with a release record. A keep on a task whose other material was gone,
# or whose one recorded thread was already accepted, answered ``already_retired`` and saved nothing, and a later pass
# bound a thread no pass had reached. A thread the task's plan record names is its material from the start:
# ``unrecorded`` until a release is recorded, never claimed gone, and nothing about the engine is claimed for it.

B, PLACE_B = "thr-2", "review_thread:thr-2"


def _second_seat(world, monkeypatch):
    """A second plan seat naming thread B, whose export read keeps failing until the test lets it through. The mock
    engine keeps one purge flag for its whole store; B is another thread, so a test resets it after A's purge."""
    def add(current):
        plan = dict(current["plan_record"])
        ask = dict(plan["asks"][0])
        ask["participants"] = [*ask["participants"], {"seat_id": "seat-b", "status": "answered", "review_thread_id": B}]
        return {**current, "plan_record": {**plan, "asks": [ask]}}

    patch_document(world.root, TASK, add)
    world.custody["runs"]["run-2"] = SimpleNamespace(run_id="run-2", task_id=TASK, invocation_id="inv-2", settled=True,
                                                     terminal_state="succeeded", continuation_of="",
                                                     started_ts="2999-01-01T00:00:00+00:00")
    named = {"inv-1": THREAD, "inv-2": B}
    monkeypatch.setattr(delegate_custody, "invocation_record",
                        lambda _root, inv, **_kw: {"request": {"_thread_id": named[inv]}} if inv in named else None)
    world.engine.runs = ["run-1", "run-2"]
    export, failing = world.engine.thread_export, {"on": True}

    def thread_export(thread_id, *, expected_store_id):
        if thread_id == B and failing["on"]:
            raise ClaudexorUnavailable("daemon_unreachable", "transient", status_code=0)
        return export(thread_id, expected_store_id=expected_store_id)

    world.engine.thread_export = thread_export
    return failing


def test_r1_a_keep_on_gone_local_material_holds_a_thread_no_release_names_yet(world):
    gone = world.root / "elsewhere" / "gone-drive"
    patch_document(world.root, TASK, lambda current: {**current, "child_drive_root": str(gone)})
    assert not world.release()  # no pass has reached the thread
    answer = owner_keep.set_owner_keep(world.root, TASK, keep=True, source=KEEP)
    assert (answer["status"], answer["keep"]) == ("partly_retired", True), answer
    assert answer["retired"] == [str(gone)] and answer["retained"] == [PLACE] and answer["unknown"] == []
    assert answer["releases"] == {PLACE: "unrecorded"}
    assert "owner_keep" in read_document(world.root, TASK)
    report = world.run()
    assert report["holds"] == {"owner_keep": 1} and world.engine.calls == [] and not world.release(), report
    assert not world.engine.purged


def test_r2_a_keep_beside_an_accepted_thread_holds_the_thread_not_reached(world, monkeypatch):
    _second_seat(world, monkeypatch)
    first = world.run()
    assert (first["accepted"], first["holds"], first["waiting"]) == (1, {"export_unavailable": 1}, 1), first
    answer = owner_keep.set_owner_keep(world.root, TASK, keep=True, source=KEEP)
    assert (answer["status"], answer["keep"]) == ("partly_retired", True), answer
    assert answer["retired"] == [PLACE] and answer["retained"] == [PLACE_B]
    assert answer["releases"] == {PLACE: "accepted_pending_removal", PLACE_B: "unrecorded"}
    assert "owner_keep" in read_document(world.root, TASK)


def test_r2b_after_that_keep_the_next_pass_neither_exports_nor_binds_the_thread_not_reached(world, monkeypatch):
    failing = _second_seat(world, monkeypatch)
    world.run()
    owner_keep.set_owner_keep(world.root, TASK, keep=True, source=KEEP)
    failing["on"], world.engine.purged, world.engine.accepted = False, False, None  # B now exports, as a live thread
    world.engine.calls.clear()
    report = world.run()
    assert report["holds"] == {"owner_keep": 1} and world.engine.calls == [], report
    assert B not in (read_document(world.root, TASK).get(rt.RELEASES) or {}) and not world.engine.purged


def test_with_no_thread_a_keep_on_wholly_gone_material_still_saves_nothing(world):
    other, gone = "task-rt-2", world.root / "elsewhere" / "gone"
    seed_result(world.root, other, document(other, "completed", child_drive_root=str(gone)))
    answer = owner_keep.set_owner_keep(world.root, other, keep=True, source=KEEP)
    assert (answer["status"], answer["keep"], answer["retired"], answer["releases"]) == (
        "already_retired", False, [str(gone)], {}), answer
    assert "owner_keep" not in read_document(world.root, other)



def test_a_stop_between_a_tasks_threads_keeps_the_rest_waiting_and_owes_the_next_pass(world, monkeypatch):
    failing = _second_seat(world, monkeypatch)
    failing["on"], stop = False, {"on": False}
    stream = world.engine.stream_thread_export_item

    def streaming(*args):  # the stop arrives while the first thread's bytes move: that thread still finishes
        stop["on"] = True
        return stream(*args)

    world.engine.stream_thread_export_item = streaming
    report = rt.retire_finished_review_threads(world.root, engine=world.engine, live=lambda _t: False, census=set(),
                                               should_stop=lambda: stop["on"])
    assert (report["accepted"], report["waiting"], report.get("stopped")) == (1, 1, True), report
    assert [thread for kind, thread in world.engine.calls if kind == "export"] == [THREAD] and B not in (
        read_document(world.root, TASK).get(rt.RELEASES) or {})
    stop["on"], world.engine.purged, world.engine.accepted = False, False, None  # B is another thread of the store
    world.engine.calls.clear()
    world.engine.stream_thread_export_item = stream
    report = world.run()
    assert report["accepted"] == 1 and "stopped" not in report, report
    assert [thread for kind, thread in world.engine.calls if kind == "export"] == [B]

def test_only_an_observed_cleanup_result_is_reported_removed(world):
    world.engine.cleaned = True  # this engine's acceptance answer already reports its cleanup done
    assert world.run()["accepted"] == 1
    answer = owner_keep.set_owner_keep(world.root, TASK, keep=True, source=KEEP)
    assert PLACE in answer["retired"] and answer["releases"] == {PLACE: "removed"}
