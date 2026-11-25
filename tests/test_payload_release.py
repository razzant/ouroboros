"""The owner's addressed release of heavy files a finished task keeps:
the mind proposes exact files with their meaning and the refs that stay, only the owner's exact by-hand choice in
Storage -> Free moves them, every proof is read again under the admission barrier and the task's custody lock,
an interruption is accounted from what the staging holds (never published unexplained), the store's derived flag
selects the tasks, and every reader says the loss without changing the accepted result's identity."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import (
    delegate_custody,
    obligations,
    owner_keep,
    review_thread_retention,
    store,
    store_import,
    subagent_worktrees,
    task_custody,
)
from ouroboros import payload_release as pr
from ouroboros import payload_release_state as release_state
from ouroboros.artifacts import actor_source_reference, store_actor_source_bytes
from ouroboros.storage_free import FreeSpace
from ouroboros.task_store import patch_document, read_document
from tests._store_shared import seed_result
from tests.test_storage_free import _guard, _join, _preview
from tests.test_store import document

TASK = "finished-1"
RUN = "delegated_runs/run1/evidence-aa"
SNAP, REPORT, SCRIPT = f"{RUN}/files/memory-snapshot.sqlite", f"{RUN}/files/REPORT.md", f"{RUN}/files/measure.py"
SECOND = f"{RUN}/files/vm-disk.img"
SUMMARY = f"{RUN}/summary.json"
CARRIER_ARGS = {"category": "delegated_activity", "source_id": "run1-1-1",
                "data": b'{"seq":1,"role":"assistant"}\n', "extension": "jsonl"}
CARRIER_REF = actor_source_reference(**CARRIER_ARGS)
CARRIER = CARRIER_REF["path"]  # published model journal, always protected
KEEP = {"basis": "test"}


@pytest.fixture
def finished(tmp_path, monkeypatch):
    """A finished task whose store holds two heavy files beside its report, script, summary and a model carrier;
    its whole work closed (the tree closure is the real reader's verdict, stood in here: ``closure["why"]`` may be
    a callable, asked each time, so a test can make the closure open only on the barrier's own view)."""
    root = tmp_path / "data"
    store_dir = root / "task_results" / "artifacts" / TASK
    for rel, data in ((SNAP, b"S" * 70_000), (SECOND, b"D" * 80_000), (REPORT, b"# measured\n"), (SCRIPT, b"print(1)\n"),
                      (SUMMARY, b"{}"), (CARRIER, CARRIER_ARGS["data"])):
        (store_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (store_dir / rel).write_bytes(data)
    seed_result(root, TASK, document(TASK, "completed", result="measured; see REPORT.md"))
    closure = {"why": "", "inside": False}

    def tree_hold(_root, task_id, _document, *, continuation_reads=None, **_kwargs):
        if continuation_reads is not None:  # the real signed read the release's line proof rechecks
            continuation_reads.load(task_id)
        return (closure["why"]() if callable(closure["why"]) else closure["why"]), {}

    monkeypatch.setattr(review_thread_retention, "tree_hold", tree_hold)
    monkeypatch.setattr(subagent_worktrees, "_task_proven_over", lambda task_id, _root, live: live(task_id) is False)
    return root, store_dir, closure


@pytest.mark.parametrize("held_kind", ["same", "child_drive", "other_task", "other_root", "none"])
def test_publication_borrows_only_the_enclosing_exact_task_custody(finished, monkeypatch, held_kind):
    from concurrent.futures import ThreadPoolExecutor

    root, directory, _ = finished
    acquire = task_custody.task_custody_lock
    monkeypatch.setattr(task_custody, "task_custody_lock",
                        lambda r, t, **kw: acquire(r, t, timeout_sec=0.01))
    held = {"same": (root, TASK),
            "child_drive": (root / "state/headless_tasks/child/data", TASK),
            "other_task": (root, "other"), "other_root": (root.parent / "other", TASK), "none": None}[held_kind]
    with acquire(root, TASK) as outer:
        assert outer
        with release_state.publication_lock(directory / SNAP, custody_held=held) as borrowed:
            assert borrowed == (held_kind in {"same", "child_drive"})
        def contender():
            with release_state.publication_lock(directory / SNAP) as locked:
                return locked
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(contender).result() is False
    with release_state.publication_lock(directory / SNAP) as acquired:
        assert acquired


def _propose(root, files=None, retained=None, consequence="the exact sample snapshot cannot be measured again"):
    files = [{"path": SNAP, "role": "frozen copy of memory.sqlite for the measurement",
              "loss": "the measurement cannot be rerun on the same bytes"}] if files is None else files
    retained = {"report": [REPORT], "method": [SCRIPT], "result": ["task_result"]} if retained is None else retained
    return pr.propose(root, TASK, files, retained, consequence, proposer="mind-1")


def _owner(live=None, guard=_guard):
    return FreeSpace(live=live or (lambda _task: False), guard=guard, offered=("stored_payloads",))


def _ready(root, owner=None):
    owner = owner or _owner()
    _preview(owner, root, root.parent)
    return owner


def _manual(owner, root):
    return owner.items(root, owner.projection(root)["plan"], "manual", 0, 500, None)["items"]


def _choose(owner, root, ids=None, accept=True):
    ids = [row["id"] for row in _manual(owner, root)] if ids is None else ids
    answer = owner.start(root, root.parent, owner.projection(root)["plan"], None, None, manual=ids, accept_loss=accept)
    _join(owner)
    return answer


def _outcome(owner, root):
    [row] = owner.items(root, owner.projection(root)["plan"], "outcomes", 0, 500, None)["items"]
    return row["outcome"], row.get("reason"), row.get("cause")


def _trash(root):
    base = root / "state" / "custody_trash"
    return sorted(base.iterdir()) if base.is_dir() else []


def _staging(root):
    base = root / "state" / "custody_staging"
    return sorted(base.iterdir()) if base.is_dir() else []


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _record(root):
    return read_document(root, TASK)[release_state.KEY]


# ---- the proposal and the store's derived flag -----------------------------------------------------------------


def test_a_proposal_writes_only_the_document_and_the_writer_derives_its_flag(finished):
    root, store_dir, _ = finished
    assert TASK not in obligations.members(root, pr.FLAG)
    answer = _propose(root)
    assert answer["ok"] and answer["files"] == [{"path": SNAP, "bytes": 70_000}]
    [offer] = _record(root)["offers"].values()
    assert offer["state"] == "open" and offer["proposed_by"] == "mind-1"
    assert [ref["role"] for ref in offer["retained"]] == ["report", "method", "result"]
    assert offer["retained"][2]["sha256"] == hashlib.sha256(b"measured; see REPORT.md").hexdigest()
    assert TASK in pr.offered(root), "the flag is the store's own column, set by the document's writer"
    with store.open_store(root, write=False) as txn:
        assert txn.row(TASK, "has_payload_offer") == 1
        assert txn.set_rows(pr.FLAG) == {}, "no owner-held row stands beside it"
    assert (store_dir / SNAP).is_file(), "a proposal moves nothing"


def test_the_flag_follows_the_document_through_release_and_its_derivation(finished):
    root, _store_dir, _ = finished
    _propose(root)
    _choose(_ready(root), root)
    assert TASK not in pr.offered(root), "accounted: nothing open, nothing pending, nothing held"
    with store.open_store(root, write=True) as txn:
        txn.conn.execute("UPDATE tasks SET has_payload_offer=NULL WHERE task_id=?", (TASK,))  # an older body's write
    store_import.upgrade_current_schema(root)  # the derivation reads the canon, never the old bit
    with store.open_store(root, write=False) as txn:
        assert txn.row(TASK, "has_payload_offer") == 0


@pytest.mark.parametrize("missing", ["column", "null"])
def test_an_older_store_gains_the_flag_once_and_derives_it_from_the_canon(finished, missing):
    root, _store_dir, _ = finished
    _propose(root)
    with store.open_store(root, write=True) as txn:
        if missing == "column":  # a store published before schema 12
            txn.conn.execute("DROP INDEX IF EXISTS tasks_has_payload_offer")
            txn.conn.execute("DROP INDEX IF EXISTS tasks_underived_flags")
            txn.conn.execute("ALTER TABLE tasks DROP COLUMN has_payload_offer")
            txn.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '11')")
        else:  # an older body's whole-row write left it underived
            txn.conn.execute("UPDATE tasks SET has_payload_offer=NULL")
    store_import.upgrade_current_schema(root)  # the first-touch writer's preparation
    with store.open_store(root, write=False) as txn:
        assert txn.row(TASK, "has_payload_offer") == 1 and not store_import._flags_need_derivation(txn)
        assert txn.meta("schema_version") == store.SCHEMA_VERSION == 12
    assert TASK in pr.offered(root)


@pytest.mark.parametrize("path, reason", [
    (CARRIER, "protected_source"), (SUMMARY, "protected_record"), ("chat_media/a.png", "protected_input"),
    ("source_handles/test-evidence/run-1/result.json", "protected_record"),
    ("source_handles/test-evidence/run-1/manifest.json", "protected_record"),
    ("source_handles/test-evidence/run-1/summary.json", "protected_record"),
    ("source_handles/test-evidence/run-1/delta.patch", "protected_record"),
    ("source_handles/test-evidence/run-1/base.pack", "protected_record"),
    ("source_handles/test-evidence/run-1/OUROBOROS_TEST_EVIDENCE_RUN.txt", "protected_record"),
    ("source_handles/tool_results/result-subject-0123456789abcdef-" + "0" * 64 + ".txt", "protected_source"),
    ("../escape", "bad_path"), ("/abs", "bad_path"), (f"{RUN}/files/none.bin", "missing"),
])
def test_protected_carriers_are_protected_in_every_subtree_and_refuse_the_whole_proposal(finished, path, reason):
    root, store_dir, _ = finished
    for rel in ("chat_media/a.png", "source_handles/test-evidence/run-1/result.json",
                "source_handles/test-evidence/run-1/manifest.json", "source_handles/test-evidence/run-1/summary.json",
                "source_handles/test-evidence/run-1/delta.patch", "source_handles/test-evidence/run-1/base.pack",
                "source_handles/test-evidence/run-1/OUROBOROS_TEST_EVIDENCE_RUN.txt",
                "source_handles/tool_results/result-subject-0123456789abcdef-" + "0" * 64 + ".txt"):
        (store_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (store_dir / rel).write_bytes(b"x")
    answer = _propose(root, files=[{"path": path, "role": "r", "loss": "l"}])
    assert not answer["ok"] and {"path": path, "reason": reason} in answer["refused"]
    assert release_state.KEY not in read_document(root, TASK)


def test_raw_test_streams_may_go_but_only_by_the_authors_exact_choice(finished):
    root, store_dir, _ = finished
    stream = "source_handles/test-evidence/run-1/stdout.bin"
    (store_dir / stream).parent.mkdir(parents=True, exist_ok=True)
    (store_dir / stream).write_bytes(b"o" * 90_000)
    assert _propose(root, files=[{"path": stream, "role": "the run's full stdout", "loss": "its exact log"}])["ok"]


def _material(store_dir, data: bytes, *, digest=None, mode=0o444):
    sha = digest or hashlib.sha256(data).hexdigest()
    rel = f"source_handles/tool_results/result-material-{sha[:16]}-{sha}.bin"
    (store_dir / rel).parent.mkdir(parents=True, exist_ok=True)
    (store_dir / rel).write_bytes(data)
    os.chmod(store_dir / rel, mode)
    return rel


def test_a_result_material_copy_is_admitted_only_by_the_identity_its_publisher_wrote(finished):
    root, store_dir, _ = finished
    good = _material(store_dir, b"M" * 66_000)
    forged = _material(store_dir, b"N" * 66_000, digest="f" * 64)  # a name the bytes do not prove
    writable = _material(store_dir, b"W" * 66_000, mode=0o644)  # not the publisher's read-only copy
    for rel in (forged, writable):
        answer = _propose(root, files=[{"path": rel, "role": "r", "loss": "l"}])
        assert {"path": rel, "reason": "result_material_unproven"} in answer["refused"]
    answer = _propose(root, files=[{"path": good, "role": "the reviewed output's exact copy", "loss": "its review cannot re-read it"}])
    assert answer["ok"]
    assert _record(root)["offers"][answer["offer"]]["files"][0]["sha256"] == hashlib.sha256(b"M" * 66_000).hexdigest()


def test_a_link_a_second_name_and_an_overlapping_or_missing_ref_are_refused(finished):
    root, store_dir, _ = finished
    os.symlink(store_dir / SNAP, store_dir / RUN / "files" / "link.sqlite")
    os.link(store_dir / REPORT, store_dir / RUN / "files" / "report-copy.md")
    linked = _propose(root, files=[{"path": f"{RUN}/files/link.sqlite", "role": "r", "loss": "l"}])
    assert {"path": f"{RUN}/files/link.sqlite", "reason": "linked"} in linked["refused"]
    shared = _propose(root, files=[{"path": f"{RUN}/files/report-copy.md", "role": "r", "loss": "l"}])
    assert {"path": f"{RUN}/files/report-copy.md", "reason": "shared"} in shared["refused"]
    overlap = _propose(root, retained={"report": [SNAP], "method": [SCRIPT], "result": ["task_result"]})
    assert any(row["reason"] == "retained_is_released" for row in overlap["refused"])
    missing = _propose(root, retained={"report": [REPORT], "method": [], "result": ["task_result"]})
    assert {"path": "retained.method", "reason": "retained_ref_required"} in missing["refused"]


def test_an_unfinished_task_cannot_be_proposed(finished):
    root, _store_dir, _ = finished
    seed_result(root, TASK, document(TASK, "running"))
    assert _propose(root) == {"ok": False, "error": "TASK_NOT_FINISHED"}


def _ctx(root, task_id="caller", **metadata):
    return SimpleNamespace(task_id=task_id, drive_root=str(root), task_metadata=metadata, task_contract={})


def test_the_tool_reads_only_what_its_caller_may_read(finished, monkeypatch):
    from ouroboros import dialogue_provenance
    from ouroboros.tools.control_storage import _propose_payload_release

    root, _store_dir, _ = finished
    args = dict(task_id=TASK, files=[{"path": SNAP, "role": "r", "loss": "l"}],
                retained={"report": [REPORT], "method": [SCRIPT], "result": ["task_result"]}, consequence="c")
    child = _propose_payload_release(_ctx(root, parent_task_id="p", root_task_id="another-tree"), **args)
    assert json.loads(child)["error"] == "ACCESS_BLOCKED", "a delegated child reads only its own tree"
    with monkeypatch.context() as patch:
        patch.setattr(dialogue_provenance, "presence_caller_binding", lambda _ctx: "binding-1")
        place = _propose_payload_release(_ctx(root), **args)
    assert json.loads(place)["error"] == "ACCESS_BLOCKED", "a Presence turn reads only its binding's work or tree"
    assert release_state.KEY not in read_document(root, TASK)
    own = _propose_payload_release(_ctx(root, "child-1", parent_task_id="p", root_task_id=TASK), **args)
    assert json.loads(own)["ok"], "its own tree, root included, is a child's to read"


# ---- what the owner sees, and who can act ------------------------------------------------------------------


def test_the_proposal_is_one_unchecked_manual_object_with_every_file_and_what_stays(finished):
    root, _store_dir, _ = finished
    _propose(root)
    owner = _ready(root)
    view = owner.projection(root)
    assert view["pool"]["count"] == 0 and view["pool"]["manual"]["count"] == 1 and view["pool"]["manual"]["payloads"] == 1
    [row] = _manual(owner, root)
    assert row["availability"] == "manual_discard" and row["kind"] == "payload" and row["bytes"] == 70_000
    manual = row["manual"]
    assert manual["payload_release"] and manual["files"][0]["role"].startswith("frozen copy")
    assert [ref["path"] for ref in manual["retained"]] == [REPORT, SCRIPT, "task_result"]


def test_cancel_no_accept_loss_an_ordinary_press_and_the_minds_free_never_release(finished):
    root, store_dir, _ = finished
    _propose(root)
    owner = _ready(root)
    [row] = _manual(owner, root)
    assert _choose(owner, root, accept=False) == {"started": False, "reason": "loss_not_accepted"}
    plan = owner.projection(root)["plan"]
    assert owner.start(root, root.parent, plan, None, [row["id"]]) == {"started": False, "reason": "not_offered"}
    assert owner.start(root, root.parent, plan, ["stored_payloads"]) == {"started": True, "reason": ""}
    _join(owner)
    assert (store_dir / SNAP).is_file(), "a whole-group press holds no proposal"
    mind = _owner()
    assert mind.request(root, root.parent, actor="task:mind", request_id="r1")["started"]
    _join(mind)
    assert (store_dir / SNAP).is_file() and not _trash(root), "the mind's Free runs only the ordinary pool"
    assert pr.release(root, {"task": TASK}, lambda _t: False, _guard, actor="mind") == ("not_owner", 0)


def test_the_owners_exact_choice_moves_exactly_the_files_and_keeps_every_ref(finished, monkeypatch):
    from ouroboros import maintenance

    root, store_dir, _ = finished
    requests = []
    monkeypatch.setattr(maintenance.child, "request", lambda: requests.append(1))
    report_sha, script_sha = _sha(store_dir / REPORT), _sha(store_dir / SCRIPT)
    _propose(root)
    owner = _ready(root)
    assert _choose(owner, root) == {"started": True, "reason": ""}
    assert not (store_dir / SNAP).exists() and requests
    assert _sha(store_dir / REPORT) == report_sha and _sha(store_dir / SCRIPT) == script_sha
    assert (store_dir / SUMMARY).is_file() and (store_dir / CARRIER).is_file() and (store_dir / SECOND).is_file()
    [entry] = _trash(root)
    assert entry.name.startswith(f"{TASK}{pr.MARK}") and sorted(p.name for p in entry.iterdir()) == ["released"]
    assert (entry / "released" / "0").stat().st_size == 70_000 and not _staging(root)
    record = _record(root)
    assert record["released"][SNAP]["bytes"] == 70_000 and list(record["offers"].values())[0]["state"] == "closed"
    run = owner.projection(root)["run"]
    assert (run["deleting"], run["deleting_bytes"]) == (1, 70_000)
    assert release_state.notice(read_document(root, TASK))["files"][0]["bytes_now"] == "unknown", "a raw document says nothing of bytes"
    assert task_custody.remove_custody_trash(root)["removed"] == 1 and not _trash(root)
    assert owner.projection(root)["run"]["removed"] == 1


# ---- proofs read again at the effect, under the barrier ---------------------------------------------------------


@pytest.mark.parametrize("change, cause", [
    (lambda root, store_dir: (store_dir / SNAP).write_bytes(b"T" * 70_001), "payload_changed"),
    (lambda root, store_dir: (store_dir / SCRIPT).write_bytes(b"print(2)\n"), "retained_changed"),
    (lambda root, store_dir: (store_dir / REPORT).unlink(), "retained_changed"),
    (lambda root, store_dir: patch_document(root, TASK, lambda current: {**current, "result": "another answer"}), "retained_changed"),
])
def test_an_input_a_retained_ref_or_the_retained_answer_changed_since_the_preview_keeps_everything(finished, change, cause):
    root, store_dir, _ = finished
    _propose(root)
    owner = _ready(root)
    change(root, store_dir)
    assert _choose(owner, root) == {"started": True, "reason": ""}
    assert _outcome(owner, root)[2] == cause and not _trash(root) and not _staging(root)
    assert (store_dir / SNAP).exists()


def test_a_keep_an_open_tree_or_a_live_task_after_the_preview_holds(finished):
    root, store_dir, closure = finished
    _propose(root)
    owner = _ready(root)
    owner_keep.set_owner_keep(root, TASK, keep=True, source=KEEP)
    assert _choose(owner, root) == {"started": True, "reason": ""}
    assert _outcome(owner, root)[2] == "owner_keep" and (store_dir / SNAP).is_file()
    owner_keep.set_owner_keep(root, TASK, keep=False, source=KEEP)
    owner = _ready(root)
    closure["why"] = "continued_tree_open"  # a Continue or an open review since the preview
    _choose(owner, root)
    assert _outcome(owner, root)[2] == "continued_tree_open" and (store_dir / SNAP).is_file()
    closure["why"] = ""
    live = {"value": False}
    owner = _ready(root, _owner(lambda _task: live["value"]))
    live["value"] = True
    _choose(owner, root)
    assert _outcome(owner, root)[2] == "not_proven_over" and (store_dir / SNAP).is_file()


def test_a_continue_claimed_after_the_outer_check_holds_on_the_barriers_own_view(finished, monkeypatch):
    """The closure is read again under the admission barrier (the queue lock a Continue's enqueue needs), and the
    task's custody lock and the probe are taken only inside it: queue -> custody, never the reverse."""
    root, store_dir, closure = finished
    seen = []
    real_lock = task_custody.task_custody_lock

    @contextlib.contextmanager
    def barrier(stop=None):
        closure["inside"] = True
        try:
            yield True
        finally:
            closure["inside"] = False

    def lock(*args, **kwargs):
        seen.append(("custody", closure["inside"]))
        return real_lock(*args, **kwargs)

    _propose(root)
    owner = _ready(root, _owner(lambda task: seen.append(("probe", closure["inside"])) or False, guard=barrier))
    monkeypatch.setattr(task_custody, "task_custody_lock", lock)
    closure["why"] = lambda: "continued_task_not_proven_over" if closure["inside"] else ""
    seen.clear()
    _choose(owner, root)
    assert _outcome(owner, root)[2] == "continued_task_not_proven_over"
    assert (store_dir / SNAP).is_file() and not _staging(root) and not _trash(root)
    assert list(_record(root)["offers"].values())[0]["state"] == "open", "refused before any decision: the offer stands"
    assert ("probe", True) in seen and not any(kind == "custody" for kind, _inside in seen), "no custody lock before the barrier said no"
    closure["why"] = ""
    seen.clear()
    owner = _ready(root, _owner(lambda task: seen.append(("probe", closure["inside"])) or False, guard=barrier))
    seen.clear()
    _choose(owner, root)
    assert not (store_dir / SNAP).exists()
    assert [inside for kind, inside in seen if kind == "custody"] == [True], "the move's custody lock is taken inside the barrier"


def test_a_held_proposal_is_named_with_its_reason_and_offers_nothing(finished):
    root, _store_dir, closure = finished
    _propose(root)
    closure["why"] = "review_custody_active"
    owner = _ready(root)
    [row] = owner.items(root, owner.projection(root)["plan"], "excluded", 0, 500, None)["items"]
    assert row["cause"] == "review_custody_active" and row["availability"] == "none"
    assert owner.projection(root)["pool"]["manual"]["count"] == 0


# ---- one release's refs are never another's files ---------------------------------------------------------------


def test_a_retained_ref_of_any_offer_is_never_released_by_another(finished):
    root, store_dir, _ = finished
    first = _propose(root)
    assert first["ok"]
    second = _propose(root, files=[{"path": REPORT, "role": "r", "loss": "l"}],
                      retained={"report": [SCRIPT], "method": [SCRIPT], "result": ["task_result"]})
    assert second == {"ok": False, "error": "RETAINED_BY_OTHER_OFFER"}
    third = _propose(root, files=[{"path": f"{RUN}/files/REPORT.md", "role": "r", "loss": "l"}])
    assert not third["ok"]
    _choose(_ready(root), root)  # the first is executed: its report, method and result stay kept for good
    after = _propose(root, files=[{"path": REPORT, "role": "r", "loss": "l"}],
                     retained={"report": [SCRIPT], "method": [SCRIPT], "result": ["task_result"]})
    assert after == {"ok": False, "error": "RETAINED_BY_OTHER_OFFER"}
    assert (store_dir / REPORT).is_file()


def test_a_file_another_open_offer_releases_cannot_be_kept_as_this_ones_ref(finished):
    root, _store_dir, _ = finished
    assert _propose(root, files=[{"path": SECOND, "role": "disk", "loss": "l"}])["ok"]
    answer = _propose(root, retained={"report": [SECOND], "method": [SCRIPT], "result": ["task_result"]})
    assert answer == {"ok": False, "error": "RETAINED_OFFERED_FOR_RELEASE"}


def _forge(root, store_dir, pid, files, retained):
    """An offer written past the proposal's own checks (an older body, a hand edit of the canon)."""
    def identity(rel):
        info = os.lstat(store_dir / rel)
        return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]

    def forge(current):
        record = dict(current.get(release_state.KEY) or {"offers": {}, "operations": {}, "released": {}})
        record["offers"] = {**record.get("offers", {}), pid: {
            "id": pid, "state": "open", "consequence": "c", "proposed_at": "t",
            "files": [{"path": rel, "role": "r", "loss": "l", "row": {}, "identity": identity(rel)} for rel in files],
            "retained": [{"role": "result", "path": "task_result", "sha256": hashlib.sha256(b"measured; see REPORT.md").hexdigest()},
                         {"role": "method", "path": SCRIPT, "identity": identity(SCRIPT)},
                         {"role": "report", "path": SCRIPT, "identity": identity(SCRIPT)},
                         *({"role": "report", "path": rel, "identity": identity(rel)} for rel in retained)]}}
        return {**current, release_state.KEY: record}

    patch_document(root, TASK, forge)


def test_two_offers_that_disagree_anyway_never_release_the_others_ref(finished):
    """The preview names such an offer held; one that became conflicting after the preview is refused by the
    move's own last look under the custody lock."""
    root, store_dir, _ = finished
    _forge(root, store_dir, "frees-report", [REPORT], [])
    owner = _ready(root)
    assert [row["id"] for row in _manual(owner, root)] == [f"payload:{TASK}{pr.MARK}frees-report"]
    _forge(root, store_dir, "keeps-report", [SECOND], [REPORT])  # written after the preview
    _choose(owner, root)
    assert _outcome(owner, root)[2] == "retained_by_other_offer" and (store_dir / REPORT).is_file()
    held = _ready(root)
    excluded = held.items(root, held.projection(root)["plan"], "excluded", 0, 500, None)["items"]
    assert any(row.get("cause") == "retained_by_other_offer" for row in excluded)


# ---- interruption, partial batches, recovery, and where the bytes are ---------------------------------------------


def test_a_failure_before_the_first_rename_moves_nothing_and_is_not_retried(finished, monkeypatch):
    root, store_dir, _ = finished
    _propose(root)
    owner = _ready(root)
    with monkeypatch.context() as patch:
        patch.setattr(pr, "_move_selected", lambda *_a: (_ for _ in ()).throw(OSError("disk")))
        _choose(owner, root)
    assert (store_dir / SNAP).is_file() and not _trash(root) and not _staging(root)
    record = _record(root)
    [operation] = record["operations"].values()
    assert operation["state"] == "accounted" and operation["moved"] == [] and not record["released"]
    assert _ready(root).projection(root)["pool"]["manual"]["count"] == 0, "the closed proposal is never retried"


def test_a_partial_batch_counts_only_what_moved_and_leaves_the_rest(finished, monkeypatch):
    root, store_dir, _ = finished
    _propose(root, files=[{"path": SNAP, "role": "snapshot", "loss": "l1"}, {"path": SECOND, "role": "disk", "loss": "l2"}])
    owner = _ready(root)
    real, calls = pr._move_selected, []

    def flaky(source, target, root, stamp):
        calls.append(source)
        if len(calls) == 2:
            raise OSError("busy")
        real(source, target, root, stamp)

    with monkeypatch.context() as patch:
        patch.setattr(pr, "_move_selected", flaky)
        _choose(owner, root)
    outcome, _reason, cause = _outcome(owner, root)
    assert (outcome, cause) == ("deleting", "release_partial")
    run = owner.projection(root)["run"]
    assert (run["deleting"], run["deleting_bytes"], run["kept"]) == (1, 70_000, []), "only the snapshot's bytes moved"
    assert not (store_dir / SNAP).exists() and (store_dir / SECOND).is_file()
    record = _record(root)
    assert list(record["released"]) == [SNAP]
    assert list(record["operations"].values())[0]["not_moved"] == [{"path": SECOND, "reason": "not_moved"}]


def test_a_stop_after_the_rename_keeps_the_staging_until_it_is_accounted_and_published(finished, monkeypatch):
    root, store_dir, _ = finished
    _propose(root)
    owner = _ready(root)
    with monkeypatch.context() as patch:
        patch.setattr(pr, "_account", lambda *_a: None)  # the process ended before its accounting write
        _choose(owner, root)
    assert _outcome(owner, root)[2] == "release_pending" and not (store_dir / SNAP).exists() and not _trash(root)
    [staged] = _staging(root)
    task_custody.sweep_custody_leftovers(root, min_age_sec=0)
    assert staged.is_dir(), "the hourly staging sweep never takes the only original"
    assert TASK in pr.offered(root), "a decided move not yet accounted keeps the task flagged"
    assert pr.recover(root)["accounted"] == [staged.name]
    assert SNAP in _record(root)["released"] and [p.name for p in _trash(root)] == [staged.name] and not _staging(root)


def test_a_staged_slot_the_decision_cannot_explain_is_held_never_published(finished, monkeypatch):
    root, store_dir, _ = finished
    _propose(root, files=[{"path": SNAP, "role": "snapshot", "loss": "l1"}, {"path": SECOND, "role": "disk", "loss": "l2"}])
    owner = _ready(root)
    with monkeypatch.context() as patch:
        patch.setattr(pr, "_account", lambda *_a: None)
        _choose(owner, root)
    [staged] = _staging(root)
    with (staged / "released" / "1").open("ab") as handle:
        handle.write(b"more")  # written after the move: no longer the bytes the owner chose
    (staged / "released" / "stray").write_bytes(b"?")  # a name the decision never made
    report = pr.recover(root)
    assert report["held"] == [staged.name]
    record = _record(root)
    [operation] = record["operations"].values()
    assert list(record["released"]) == [SNAP]
    assert sorted(row["reason"] for row in operation["staging_held"]) == ["staged_changed", "staging_unknown"]
    [entry] = _trash(root)
    assert sorted(p.name for p in (entry / "released").iterdir()) == ["0"], "only the explained slot is handed over"
    assert sorted(p.name for p in (staged / "released").iterdir()) == ["1", "stray"], "the rest stays where it is"
    assert TASK in pr.offered(root)
    held = _ready(root)
    excluded = held.items(root, held.projection(root)["plan"], "excluded", 0, 500, None)["items"]
    assert any(row.get("cause") == "release_unexplained" for row in excluded), "Storage names it held"


def test_a_decision_whose_staging_never_appeared_is_accounted_as_nothing_moved(finished, monkeypatch):
    root, store_dir, _ = finished
    _propose(root)
    owner = _ready(root)
    with monkeypatch.context() as patch:
        patch.setattr(pr, "_account", lambda *_a: None)
        patch.setattr(pr, "_move_selected", lambda *_a: (_ for _ in ()).throw(OSError("stopped")))
        _choose(owner, root)
    for staged in _staging(root):
        (staged / "released").rmdir()
        staged.rmdir()  # the stop came before the staging was made
    pr.recover(root)
    [operation] = _record(root)["operations"].values()
    assert operation["state"] == "accounted" and operation["not_moved"] == [{"path": SNAP, "reason": "not_moved"}]
    assert (store_dir / SNAP).is_file() and TASK not in pr.offered(root)


def test_handoff_waits_when_the_staged_metadata_read_is_refused(finished, monkeypatch):
    root, _store_dir, _ = finished
    _propose(root)
    owner = _ready(root)
    with monkeypatch.context() as patch:
        patch.setattr(pr, "_publish", lambda *_a: (_ for _ in ()).throw(OSError("stopped")))
        _choose(owner, root)  # accounted, not yet handed over
    [staged] = _staging(root)
    slot, real = staged / "released" / "0", os.lstat
    original = slot.read_bytes()

    def refuse(path, *args, **kwargs):
        if Path(path) == slot:
            raise OSError(5, "synthetic temporary refusal")
        return real(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "lstat", refuse)
        report = pr.recover(root)
    assert len(report["errors"]) == 1 and not _trash(root) and slot.read_bytes() == original
    assert SNAP in _record(root)["released"]
    pr.recover(root)
    assert [p.name for p in _trash(root)] == [staged.name] and not _staging(root)


def test_a_typed_custody_refusal_for_one_task_does_not_stop_the_next(finished, monkeypatch):
    from ouroboros import task_store

    root, _store_dir, _ = finished
    operation = {"op1": {"state": "moving", "staging": "none"}}
    accounted = []

    @contextlib.contextmanager
    def fake_gate(_root, task_id, *, timeout_sec=2.0):
        if task_id == "task-a":
            raise task_custody.CustodyBusy("synthetic temporary custody refusal")
        yield ""

    monkeypatch.setattr(owner_keep, "gate", fake_gate)
    monkeypatch.setattr(pr, "offered", lambda _root: {"task-a": {}, "task-b": {}})
    monkeypatch.setattr(task_store, "read_document", lambda *_a, **_k: {})
    monkeypatch.setattr(pr, "release_record", lambda _doc: {"operations": operation})
    monkeypatch.setattr(pr, "_account", lambda _root, task_id, op, _staging: accounted.append((task_id, op)) or {})
    report = pr.recover(root)
    assert report["errors"] == ["task-a:op1: CustodyBusy"] and report["accounted"] == ["task-b:op1"]
    assert accounted == [("task-b", "op1")]


def test_a_file_named_by_the_decision_and_absent_everywhere_is_unknown_missing_not_released(finished, monkeypatch):
    from ouroboros.gateway.task_archive import serve_task_file

    root, store_dir, _ = finished
    _propose(root)
    owner = _ready(root)
    with monkeypatch.context() as patch:
        patch.setattr(pr, "_account", lambda *_a: None)
        _choose(owner, root)
    [staged] = _staging(root)
    (staged / "released" / "0").unlink()  # something outside the protocol took it
    pr.recover(root)
    record = _record(root)
    assert not record["released"]
    assert list(record["operations"].values())[0]["not_moved"] == [{"path": SNAP, "reason": "unknown_missing"}]
    (store_dir / SNAP).write_bytes(b"stale bytes after an unaccounted loss")
    response = serve_task_file(root, store_dir, SNAP, Path(SNAP).name, None, task_id=TASK)
    assert response.status_code == 503 and json.loads(response.body)["reason_code"] == "artifact_unavailable"
    assert release_state.keep_places(root, TASK, read_document(root, TASK), [str(store_dir / SNAP)])[str(store_dir / SNAP)] == ("unknown", None)


def test_readers_say_where_the_bytes_are_staged_deleting_or_deleted(finished, monkeypatch):
    from ouroboros.task_status import load_effective_task_result

    root, _store_dir, _ = finished
    _propose(root)
    owner = _ready(root)

    def bytes_now():
        rows = load_effective_task_result(root, TASK)["artifacts"]
        address = Path(SNAP).parts
        return next(row["released"]["bytes_now"] for row in rows
                    if Path(str(row.get("path") or "")).parts[-len(address):] == address)

    with monkeypatch.context() as patch:
        patch.setattr(pr, "_publish", lambda *_a: (_ for _ in ()).throw(OSError("full")))
        _choose(owner, root)
    assert _outcome(owner, root)[2] == "release_pending" and bytes_now() == "staged"
    pr.recover(root)
    assert bytes_now() == "deleting"
    task_custody.remove_custody_trash(root)
    assert bytes_now() == "deleted"


# ---- the compressor's final replace -----------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "darwin", reason="the replace race is the macOS decmpfs path")
def test_the_compressor_never_publishes_a_file_back_after_the_release(finished):
    from ouroboros import artifact_storage

    root, store_dir, _ = finished
    path = store_dir / SNAP
    before = os.lstat(path)

    def serial():
        return task_custody.task_custody_lock(root, TASK, timeout_sec=0)

    with task_custody.task_custody_lock(root, TASK, timeout_sec=1):  # the release holds it
        assert artifact_storage._compress_macos(path, before, None, False, serial) == "busy"

    def released_first():  # the owner's release took the lock first and moved the file after the hash
        os.rename(path, store_dir / "moved-away")
        return serial()

    outcome = artifact_storage.compression_outcome(path, serial=released_first)
    if outcome in ("shared", "share_unknown", "incompressible", "dataless", "linked", "low_space"):
        pytest.skip(f"this volume's floor kept the file before the race: {outcome}")
    assert outcome == "failed" and not path.exists(), "the compressed sibling never takes the released name"


# ---- the accepted result keeps its identity, for every kind of source --------------------------------------------


def _every_kind_of_source(root, store_dir):
    """A recorded top-level row, a registered immutable top-level file, a nested recorded immutable capture, an
    unrecorded nested evidence listing and a file of a task's own retained evidence generation."""
    recorded, registered = b"R" * 66_000, b"G" * 66_000
    nested, generation = b"C" * 66_000, b"E" * 66_000
    for rel, data in (("recorded.bin", recorded), ("registered.bin", registered), ("captures/c1/out.bin", nested),
                      ("named_evidence/files/.review-evidence/big.bin", generation)):
        (store_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (store_dir / rel).write_bytes(data)
    (store_dir / ".artifact_manifest.json").write_text(json.dumps({"artifacts": {"registered.bin": {
        "kind": "user_file", "immutable": True, "size": len(registered), "sha256": hashlib.sha256(registered).hexdigest()}}}))
    rows = [{"kind": "task_artifact", "name": "recorded.bin", "path": str(store_dir / "recorded.bin"), "status": "ready",
             "size": len(recorded), "sha256": hashlib.sha256(recorded).hexdigest(), "errors": []},
            {"kind": "workspace_capture", "name": "out.bin", "relpath": "captures/c1/out.bin", "immutable": True,
             "path": str(store_dir / "captures/c1/out.bin"), "status": "ready", "size": len(nested),
             "sha256": hashlib.sha256(nested).hexdigest(), "errors": []}]
    seed_result(root, TASK, document(TASK, "completed", result="measured; see REPORT.md", artifacts=rows))
    return ["recorded.bin", "registered.bin", "captures/c1/out.bin", SNAP, "named_evidence/files/.review-evidence/big.bin"]


def test_the_release_never_changes_the_accepted_results_hash_for_any_kind_of_source(finished):
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools.join_ledger import _child_result_sha256

    root, store_dir, _ = finished
    paths = _every_kind_of_source(root, store_dir)
    before = load_effective_task_result(root, TASK)
    sha = _child_result_sha256(before)
    assert _propose(root, files=[{"path": rel, "role": "r", "loss": "l"} for rel in paths])["ok"]
    _choose(_ready(root), root)
    after = load_effective_task_result(root, TASK)
    assert _child_result_sha256(after) == sha, "a loss receipt never reopens an accepted disposition"
    assert after.get("artifact_status") == before.get("artifact_status")
    released = {row.get("relpath") or row.get("name"): row for row in after["artifact_bundle"]["artifacts"] if row.get("released")}
    assert set(released) == set(paths) and {row["status"] for row in released.values()} == {"released"}, "never ready, never missing"
    forged = dict(after, artifacts=[dict(row, released={"status_before": "ready"}) for row in after["artifacts"]
                                   if not row.get("released")] + [row for row in after["artifacts"] if row.get("released")])
    forged["payload_release"] = {}
    assert _child_result_sha256(forged) != sha, "a released fact the canon does not hold changes the hash"


def test_the_archive_never_serves_a_released_member_and_counts_it_apart(finished):
    from ouroboros.gateway.task_archive import _archive_members, directory_archives
    from ouroboros.task_status import load_effective_task_result

    root, store_dir, _ = finished
    _propose(root)
    _choose(_ready(root), root)
    rows = load_effective_task_result(root, TASK)["artifacts"]
    stores = task_custody.task_artifact_stores(root, TASK)
    view = directory_archives(stores, rows, anchor=root)
    assert view["delegated_runs"]["released"] == 1 and view["delegated_runs"]["files"] >= 2
    (store_dir / SNAP).write_bytes(b"a new file under the released name")  # never a member again
    members = [name for name, _route, _row in _archive_members(stores, rows, "delegated_runs", anchor=root)]
    assert not any(name.endswith("memory-snapshot.sqlite") for name in members)


@pytest.mark.parametrize("borrowed", [False, True])
def test_a_reused_named_evidence_generation_publishes_what_it_still_holds(finished, monkeypatch, borrowed):
    """The real publisher path: ``retain_named`` reuses an unchanged generation and its custody receipt says how
    many files the owner released; the immutable summary on disk is never rewritten."""
    from ouroboros import delegate_named_evidence as dne
    from ouroboros.delegate_capture_attempt import POLICY

    root, store_dir, _ = finished
    area = root.parent / "copy"
    (area / ".review-evidence").mkdir(parents=True)
    entry = SimpleNamespace(task_id=TASK, run_id="run1", capture_id="", snapshot_id="run1", invocation_id="inv",
                            baseline_sha="b", terminal_source={"sha256": "t"}, execution_root=str(area),
                            named_evidence={"directory": "evidence-aa", "status": "retained"})
    source = {"policy": POLICY, "limit_bytes": None, "whole": False, "baseline_sha": "b", "terminal_sha": "t",
              "invocation_id": "inv", "snapshot_id": "run1", "capture_id": "", "witness": dne.named_evidence_witness(area),
              "product_selection": hashlib.sha256(json.dumps([]).encode()).hexdigest()}
    summary = {"status": "retained", "files": [{"path": "memory-snapshot.sqlite", "size": 70_000, "sha256": "x", "mode": 420},
                                               {"path": "REPORT.md", "size": 11, "sha256": "y", "mode": 420}],
               "gaps": [], "inputs": source, "witness": source["witness"], "evidence_dir": str(store_dir / RUN)}
    (store_dir / SUMMARY).write_text(json.dumps(summary))
    on_disk = (store_dir / SUMMARY).read_bytes()
    emitted = []
    monkeypatch.setattr(delegate_custody, "emit", lambda _root, kind, row: emitted.append((kind, row)) or True)
    _propose(root)
    _choose(_ready(root), root)
    with task_custody.task_custody_lock(root, TASK) if borrowed else contextlib.nullcontext(True) as held:
        assert held
        answer = dne.retain_named(root, entry, custody_held=(root, TASK) if borrowed else None)
    assert answer["files"] == [{"path": "REPORT.md", "size": 11, "sha256": "y", "mode": 420}]
    assert answer["released"] == ["files/memory-snapshot.sqlite"]
    [(kind, row)] = emitted
    assert kind == delegate_custody.NAMED_EVIDENCE_RETAINED and row["outcome"]["released"] == 1
    assert (store_dir / SUMMARY).read_bytes() == on_disk
    assert dne.saved_named_evidence(root, entry, observe=False)[1]["released"] == ["files/memory-snapshot.sqlite"]
    (area / ".review-evidence" / "new-output.txt").write_text("new observation", encoding="utf-8")
    changed = dne.retain_named(root, entry, recapture=True)
    assert changed["status"] == "partial" and changed["reason"] == "released_generation_changed"
    assert (store_dir / SUMMARY).read_bytes() == on_disk and len(emitted) == 1


# ---- releases against a finished task and its actual custody records ----

@pytest.fixture
def real_finished(tmp_path):
    """Real task/queue/custody/Continue readers; no stub of tree_hold or terminal proof."""
    from ouroboros.task_results import write_task_result
    from ouroboros.utils import utc_now_iso
    from tests._shared import write_queue_document

    root = tmp_path / "data"
    store_dir = release_state.store_of(root, TASK)
    for rel, data in ((SNAP, b"S" * 70_000), (REPORT, b"# synthetic report\n"),
                      (SCRIPT, b"print('synthetic')\n")):
        path = store_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    assert store_actor_source_bytes(root, TASK, **CARRIER_ARGS) == CARRIER_REF
    write_task_result(root, TASK, "completed", result="synthetic measurement complete",
                      root_task_id=TASK, delegation_role="root", reason_code="final_message")
    write_queue_document(root / "state" / "queue_snapshot.json", {"ts": utc_now_iso(), "pending": [], "running": []})
    assert pr.holds(root, TASK, read_document(root, TASK), lambda _task: False) == ""
    return root, store_dir


def test_author_stop_real_continue_offer_holds_release_then_finished_work_is_offered(real_finished):
    from ouroboros.owner_continue import continuation_offer

    root, store_dir = real_finished
    assert _propose(root)["ok"]
    patch_document(root, TASK, lambda row: {**row, "outcome_axes": {"execution": {
        "status": "ok", "task_completion": {"action": "stop", "rationale": "Need another investigation."}}}})
    assert continuation_offer(read_document(root, TASK), TASK)["cause"] == "author_stop"
    assert pr.candidates(root, lambda _task: False)[0]["hold"] == "continuable"
    owner = _ready(root)
    assert _manual(owner, root) == [] and (store_dir / SNAP).exists()
    patch_document(root, TASK, lambda row: {**row, "outcome_axes": {}})
    assert _choose(_ready(root), root)["started"]
    assert not (store_dir / SNAP).exists()


def test_patch_verdict_writer_and_collision_family_are_never_proposed(real_finished):
    from ouroboros.tools.patch_verdict import write_patch_verdict, retained_verdicts

    root, store_dir = real_finished
    ctx = SimpleNamespace(drive_root=root, task_id=TASK)
    for reason in ("First decision", "Corrected decision"):
        write_patch_verdict(ctx, "synthetic-child", outcome="rejected", reason=reason, files=[], manifest={},
                            applied=False, conflicts=[], protected=[])
    assert len(list(retained_verdicts(root, TASK, "synthetic-child"))) == 2
    paths = sorted(store_dir.glob("subagent_patch_verdict_*.json"))
    assert len(paths) == 2
    for path in paths:
        answer = _propose(root, files=[{"path": path.name, "role": "mistaken payload", "loss": "lose decision"}])
        assert answer["refused"] == [{"path": path.name, "reason": "protected_record"}]
    assert _propose(root)["ok"]  # the actual measurement snapshot remains a legitimate proposal


def test_replace_and_withdraw_keep_old_record_and_require_new_owner_choice(real_finished):
    root, store_dir = real_finished
    first = _propose(root)["offer"]
    old = json.loads(json.dumps(_record(root)["offers"][first]))
    owner = _ready(root)
    old_selection = _manual(owner, root)[0]["id"]
    source = store_dir / SNAP
    changed = source.with_suffix(".replacement")
    changed.write_bytes(source.read_bytes())
    os.replace(changed, source)  # byte-preserving publication still invalidates the inode-bound offer
    assert pr.candidates(root, lambda _task: False)[0]["hold"] == "payload_changed"
    replacement = pr.propose(root, TASK, [{"path": SNAP, "role": "snapshot", "loss": "same measurement lost"}],
                             {"report": REPORT, "method": SCRIPT, "result": "task_result"}, "new choice", replace_offer=first)
    assert replacement["ok"]
    record = _record(root)
    assert record["offers"][first] == {**old, "state": "superseded", "replaced_by": replacement["offer"],
                                         "closed_at": record["offers"][first]["closed_at"]}
    _choose(owner, root, [old_selection])
    assert source.exists() and _outcome(owner, root)[2] == "changed"
    second = replacement["offer"]
    assert pr.propose(root, TASK, None, None, "", withdraw_offer=second)["withdrawn"] == second
    assert _record(root)["offers"][second]["state"] == "withdrawn" and source.exists()
    assert not pr.candidates(root, lambda _task: False)
    assert _propose(root)["ok"] and _choose(_ready(root), root)["started"]
    assert not source.exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="real decmpfs publication needs macOS")
def test_real_compression_publication_can_be_reproposed(real_finished):
    from ouroboros.artifact_storage import compression_outcome

    root, store_dir = real_finished
    first = _propose(root)["offer"]
    result = compression_outcome(store_dir / SNAP, serial=lambda: task_custody.task_custody_lock(root, TASK))
    if result != "compressed":
        pytest.skip(f"volume did not compress the synthetic snapshot: {result}")
    assert pr.candidates(root, lambda _task: False)[0]["hold"] == "payload_changed"
    fresh = pr.propose(root, TASK, [{"path": SNAP, "role": "snapshot", "loss": "measurement lost"}],
                       {"report": REPORT, "method": SCRIPT, "result": "task_result"}, "same loss", replace_offer=first)
    assert fresh["ok"] and fresh["offer"] != first
    assert _choose(_ready(root), root)["started"] and not (store_dir / SNAP).exists()


@pytest.mark.parametrize("failure", ["account", "publish"])
def test_recovery_reconciles_the_same_free_result_and_wakes_trash(real_finished, monkeypatch, failure):
    from ouroboros import maintenance

    root, store_dir = real_finished
    _propose(root)
    owner = _ready(root)
    with monkeypatch.context() as patch:
        if failure == "account":
            patch.setattr(pr, "_account", lambda *_args: None)
        else:
            patch.setattr(pr, "_publish", lambda *_args: (_ for _ in ()).throw(OSError("publish failed")))
        _choose(owner, root)
    assert _outcome(owner, root)[2] == "release_pending" and not (store_dir / SNAP).exists()
    requests = []
    monkeypatch.setattr(maintenance.child, "request", lambda: requests.append(True))
    assert pr.recover(root)["accounted"] and requests
    assert _outcome(owner, root)[0] == "deleting"
    view = owner.projection(root)["run"]
    assert view["deleting_bytes"] == 70_000 and view["kept"] == [] and view["not_started"] == 0
    task_custody.remove_custody_trash(root)
    assert _outcome(owner, root)[0] == "removed"
    assert owner.projection(root)["run"]["removed_bytes"] == 70_000
    assert len(_record(root)["operations"]) == 1


@pytest.mark.parametrize("error", [OSError(5, "temporary EIO"), PermissionError(13, "temporary access failure")])
def test_unreadable_staged_slot_is_retryable_not_accounted_missing(real_finished, monkeypatch, error):
    root, _store_dir = real_finished
    _propose(root)
    owner = _ready(root)
    with monkeypatch.context() as patch:
        patch.setattr(pr, "_account", lambda *_args: None)
        _choose(owner, root)
    [staged] = _staging(root)
    slot, original = staged / "released" / "0", os.lstat

    def failed(path, *args, **kwargs):
        if Path(path) == slot:
            raise error
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(pr.os, "lstat", failed)
        assert not pr.recover(root)["accounted"]
    [operation] = _record(root)["operations"].values()
    assert operation["state"] == "moving" and TASK in pr.offered(root)
    assert not _record(root)["released"] and slot.exists()
    assert pr.recover(root)["accounted"]
    assert SNAP in _record(root)["released"] and _outcome(owner, root)[0] == "deleting"


def test_one_malformed_proposal_does_not_hide_healthy_proposals(real_finished):
    root, _store_dir = real_finished
    _propose(root)
    def corrupt(row):
        row[release_state.KEY]["offers"]["broken"] = {"state": "open", "files": [None]}
        return row
    patch_document(root, TASK, corrupt)
    found = pr.candidates(root, lambda _task: False)
    assert len(found) == 2
    assert {entry["hold"] for entry in found} == {"", "proposal_unreadable"}
    assert len(_manual(_ready(root), root)) == 1


def test_consequences_are_retained_whole_and_released_refs_never_reappear_as_retained(real_finished):
    root, store_dir = real_finished
    text = "scope " * 150 + "\nThe last sentence changes the loss."
    _propose(root, consequence=text)
    [offer] = _record(root)["offers"].values()
    assert offer["consequence"] == text
    _choose(_ready(root), root)
    (store_dir / SNAP).write_bytes(b"reappeared copy")
    (store_dir / SECOND).write_bytes(b"different material")
    answer = _propose(root, files=[{"path": SECOND, "role": "disk", "loss": "disk gone"}],
                      retained={"report": SNAP, "method": SCRIPT, "result": "task_result"})
    assert answer["refused"] == [{"path": f"retained.report:{SNAP}", "reason": "retained_is_released"}]


@pytest.mark.skipif(os.name == "nt", reason="retained downloads need directory-relative no-follow opens")
def test_real_storage_api_exact_choice_preserves_retained_bytes(real_finished):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient
    from ouroboros.gateway.storage import storage_routes
    from ouroboros.gateway.tasks import api_task_artifact

    root, store_dir = real_finished
    kept = {rel: (store_dir / rel).read_bytes() for rel in (REPORT, SCRIPT, CARRIER)}
    assert _propose(root)["ok"]
    owner = _owner()
    app = Starlette(routes=[*storage_routes(root, repo_dir=root.parent, free_owner=owner),
        Route("/api/tasks/{task_id}/artifacts/{name}", endpoint=api_task_artifact)])
    app.state.drive_root = root

    def download(client, rel, selector):
        return client.get(f"/api/tasks/{TASK}/artifacts/{Path(rel).name}", params={selector: rel})

    def assert_retained(client):
        for rel, expected in kept.items():
            selector = "source" if rel == CARRIER else "relpath"
            response = download(client, rel, selector)
            assert response.status_code == 200, (rel, selector, response.text)
            assert response.content == expected, rel
            if rel == CARRIER:
                assert response.headers["x-ouroboros-artifact-identity"] == "verified"
                assert response.headers["x-ouroboros-artifact-sha256"] == CARRIER_REF["sha256"]
        assert download(client, CARRIER, "relpath").status_code == 404

    with TestClient(app) as client:
        assert_retained(client)
        assert download(client, SNAP, "relpath").status_code == 200
        assert client.post("/api/storage/free/preview").json()["started"]
        _join(owner)
        ready = client.get("/api/storage").json()["free"]
        query = {"plan": ready["plan"], "view": "manual", "cursor": 0, "limit": 500}
        [item] = client.get("/api/storage/free/items", params=query).json()["items"]
        assert {row["path"] for row in item["manual"]["retained"]} == {REPORT, SCRIPT, "task_result"}
        selection = {"plan": ready["plan"], "manual": [item["id"]]}
        assert client.post("/api/storage/free", json=selection).json()["reason"] == "loss_not_accepted"
        assert (store_dir / SNAP).exists()
        assert client.post("/api/storage/free", json={**selection, "accept_loss": True}).json()["started"]
        _join(owner)
        response = download(client, SNAP, "relpath")
        assert response.status_code == 410 and response.json()["reason_code"] == "artifact_released"
        assert_retained(client)
    assert all((store_dir / rel).read_bytes() == expected for rel, expected in kept.items())


def _dual_store(real_finished):
    """A registered immutable file present in both the canonical and own execution store."""
    from ouroboros import artifacts
    from ouroboros.task_results import write_task_result

    root, canonical = real_finished
    drive = root / "task_drives" / TASK
    source = root.parent / "synthetic-payload.bin"
    source.write_bytes(b"synthetic measurement bytes\n")
    child_row = artifacts.copy_file_to_task_artifacts(SimpleNamespace(drive_root=drive, task_id=TASK), source, immutable=True)
    canonical_row = artifacts.copy_file_to_task_artifacts(SimpleNamespace(drive_root=root, task_id=TASK), source, immutable=True)
    write_task_result(drive, TASK, "running", artifacts=[child_row])
    patch_document(root, TASK, lambda row: {**row, "child_drive_root": str(drive), "artifacts": [canonical_row]})
    return root, canonical, drive, child_row, canonical_row


@pytest.mark.parametrize("variant", ["same", "changed", "unproved", "released_collision"])
def test_real_release_child_settlement_reader_download_chain(real_finished, variant):
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools.join_ledger import _child_result_sha256
    from tests.test_task_file_serving import _client

    root, canonical, drive, child_row, canonical_row = _dual_store(real_finished)
    name = canonical_row["name"]
    files = [{"path": name, "role": "synthetic data", "loss": "exact rerun lost"}]
    changed_bytes = b"a different surviving result"
    if variant == "released_collision":
        conflict = task_custody._collision(canonical / name, hashlib.sha256(changed_bytes).hexdigest())
        conflict.write_bytes(changed_bytes)
        files.append({"path": conflict.name, "role": "previous conflicting data", "loss": "old conflict lost"})
    before = _child_result_sha256(load_effective_task_result(root, TASK))
    assert _propose(root, files=files)["ok"]
    _choose(_ready(root), root)
    assert not (canonical / name).exists() and Path(child_row["path"]).exists()
    receipt = _record(root)["released"][name]
    for directory in (canonical, release_state.store_of(drive, TASK)):
        assert release_state.states_under(directory)[name] == receipt
    from ouroboros.gateway.task_archive import serve_task_file
    assert serve_task_file(root, release_state.store_of(drive, TASK), name, name, None, task_id=TASK).status_code == 410
    with _client(root) as client:
        assert client.get(f"/api/tasks/{TASK}/artifacts/{name}").status_code == 410
    projected = load_effective_task_result(root, TASK)
    assert _child_result_sha256(projected) == before
    assert next(row for row in projected["artifacts"] if row.get("name") == name)["released"]
    if variant in {"changed", "released_collision"}:
        Path(child_row["path"]).chmod(0o644)
        Path(child_row["path"]).write_bytes(changed_bytes)
    elif variant == "unproved":
        def old_receipt(row):
            row[release_state.KEY]["released"][name].pop("sha256")
            return row
        patch_document(root, TASK, old_receipt)
    remaining = Path(child_row["path"]).read_bytes()
    outcome = task_custody.settle_child_drive(root, TASK, drive, live=lambda _task: False)
    assert outcome["status"] == "removed", outcome
    assert not drive.exists() and not (canonical / name).exists()
    original = canonical / name
    conflicts = list(canonical.glob(f"{original.stem}.*{original.suffix}"))
    if variant in {"same", "released_collision"}:
        assert conflicts == []
    else:
        [conflict] = conflicts
        assert conflict.read_bytes() == remaining
    if variant == "same":
        assert _child_result_sha256(load_effective_task_result(root, TASK)) == before
    # A stale publisher's reappeared canonical file must not bypass the release via immutable fast path.
    (canonical / name).write_bytes(remaining)
    with _client(root) as client:
        response = client.get(f"/api/tasks/{TASK}/artifacts/{name}")
        assert response.status_code == 410 and response.json()["reason_code"] == "artifact_released"
    assert next(row for row in load_effective_task_result(root, TASK)["artifacts"] if row.get("name") == name)["released"]


@pytest.mark.parametrize("changed", [False, True])
def test_copyback_respects_release_and_preserves_conflicting_child(real_finished, changed):
    from ouroboros.headless import _copy_child_artifacts_to_parent

    root, canonical, drive, child_row, canonical_row = _dual_store(real_finished)
    name = canonical_row["name"]
    _propose(root, files=[{"path": name, "role": "snapshot", "loss": "exact rerun lost"}])
    _choose(_ready(root), root)
    if changed:
        Path(child_row["path"]).chmod(0o644)
        Path(child_row["path"]).write_bytes(b"changed child")
    promotion = {"pending_refs": []}
    with task_custody.task_custody_lock(root, TASK):
        rows = _copy_child_artifacts_to_parent(root, TASK, drive, [child_row], promotion=promotion)
    assert not (canonical / name).exists() and Path(child_row["path"]).exists()
    if changed:
        assert "released_path_conflict" in promotion["pending_refs"][0]["reason"]
    else:
        assert not promotion["pending_refs"] and rows[0]["path"] == str(canonical / name)


@pytest.mark.parametrize("changed", [False, True])
def test_raw_evidence_settlement_respects_release(real_finished, changed):
    from ouroboros.task_results import write_task_result

    root, canonical = real_finished
    drive = root / "task_drives" / TASK
    rel = "source_handles/test-evidence/synthetic-run/large.bin"
    source = release_state.store_of(drive, TASK) / rel
    source.parent.mkdir(parents=True)
    source.write_bytes(b"raw evidence bytes")
    target = canonical / rel
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())
    write_task_result(drive, TASK, "running")
    patch_document(root, TASK, lambda row: {**row, "child_drive_root": str(drive)})
    _propose(root, files=[{"path": rel, "role": "raw test data", "loss": "old run not reproducible"}])
    _choose(_ready(root), root)
    if changed:
        source.write_bytes(b"different raw evidence")
    outcome = task_custody.settle_child_drive(root, TASK, drive, live=lambda _task: False)
    assert not target.exists()
    if changed:
        assert outcome["status"] == "retained" and source.read_bytes() == b"different raw evidence"
        pending = read_document(root, TASK)["child_ref_promotion"]["pending_refs"]
        assert any("released_path_conflict" in row["reason"] for row in pending)
    else:
        assert outcome["status"] == "removed" and not drive.exists()


def test_named_evidence_reappearing_path_and_result_material_do_not_claim_preserved(real_finished):
    from ouroboros.delegate_named_evidence import said_released
    from ouroboros.result_material import _retained_copy

    root, canonical = real_finished
    _propose(root)
    _choose(_ready(root), root)
    (canonical / SNAP).write_bytes(b"stale reappearance")
    summary = {"evidence_dir": str(canonical / RUN), "files": [{"path": "memory-snapshot.sqlite"}, {"path": "REPORT.md"}]}
    assert said_released(summary)["files"] == [{"path": "REPORT.md"}]
    item = {"path": str(canonical / SNAP), "size": 18, "sha256": _sha(canonical / SNAP)}
    assert "retained_gap" in _retained_copy(root, TASK, item, {"immutable": True, "sha256": item["sha256"]})


def test_failed_rename_keeps_errno_in_the_same_free_outcome(real_finished, monkeypatch):
    root, canonical = real_finished
    _propose(root)
    owner = _ready(root)
    def refuse(*_args):
        raise OSError(5, "synthetic EIO")
    with monkeypatch.context() as patch:
        patch.setattr(pr, "_move_selected", refuse)
        _choose(owner, root)
    outcome, _reason, cause = _outcome(owner, root)
    assert (outcome, cause) == ("failed", "release_failed:5")
    assert owner.projection(root)["run"]["failed"] == 1
    assert (canonical / SNAP).exists() and not _record(root)["released"]
    pr.recover(root)
    assert _outcome(owner, root)[2] == "release_failed:5"
    assert len(_record(root)["operations"]) == 1


def test_projection_during_publication_does_not_count_the_choice_twice(real_finished, monkeypatch):
    root, _canonical = real_finished
    _propose(root)
    owner, publish = _ready(root), pr._publish
    def observed(*args):
        owner.projection(root)  # the operation exists, but the worker has not classified the choice yet
        return publish(*args)
    monkeypatch.setattr(pr, "_publish", observed)
    _choose(owner, root)
    for _ in range(2):
        view = owner.projection(root)["run"]
        assert (view["selected"], view["deleting"], view["deleting_bytes"]) == (1, 1, 70_000)


@pytest.mark.parametrize("rel", ["/absolute.bin", "//server/share/file", r"C:\\absolute.bin", "C:/file", "D:file",
                                "ordinary/C:/file", "ordinary/d:file", r"\rooted", r"ordinary\file", "a//b",
                                "a/./b", "a/../b", "a\x00b"])
def test_platform_qualified_or_ambiguous_spelling_is_refused_for_files_and_refs(finished, rel):
    root, store_dir, _ = finished
    answer = _propose(root, files=[{"path": rel, "role": "raw", "loss": "rerun"}])
    assert {"path": rel, "reason": "bad_path"} in answer["refused"]
    answer = _propose(root, retained={"report": rel, "method": SCRIPT, "result": "task_result"})
    assert {"path": f"retained.report:{rel}", "reason": "bad_path"} in answer["refused"]
    assert release_state.KEY not in read_document(root, TASK) and (store_dir / REPORT).is_file()


@pytest.mark.parametrize("level", ["store", "artifacts"])
def test_link_or_junction_in_fixed_store_ancestry_refuses_offer_and_confirmation(finished, level):
    root, store_dir, _ = finished
    assert _propose(root)["ok"]
    owner = _ready(root)
    entry = store_dir if level == "store" else store_dir.parent
    moved = root.parent / "other-directory"
    entry.rename(moved)
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(moved), str(entry))
    else:
        entry.symlink_to(moved, target_is_directory=True)
    assert not _propose(root)["ok"]
    _choose(owner, root)
    assert not _record(root)["operations"]
    assert (store_dir / SNAP).read_bytes() == b"S" * 70_000 and (store_dir / REPORT).is_file()


@pytest.mark.parametrize("style", ["case", "normalization"])
def test_native_alias_cannot_release_any_retained_role_even_if_an_old_offer_uses_it(finished, style):
    from unicodedata import normalize

    root, store_dir, _ = finished
    exact = "Run-Évidence.bin"
    (store_dir / exact).write_bytes(b"one-link evidence retained by the owner")
    # Enumeration is the durable spelling, including filesystems that return a decomposed name.
    exact = next(name for name in os.listdir(store_dir) if normalize("NFC", name) == exact)
    alias = exact.swapcase() if style == "case" else normalize("NFD" if exact == normalize("NFC", exact) else "NFC", exact)
    if not (store_dir / alias).exists() or not (store_dir / alias).samefile(store_dir / exact):
        pytest.skip("the directory does not accept this native alias")
    assert (store_dir / exact).stat().st_nlink == 1
    file = {"path": exact, "role": "raw", "loss": "exact source"}
    for role in ("report", "method", "result"):
        retained = {"report": REPORT, "method": SCRIPT, "result": "task_result", role: alias}
        assert not _propose(root, files=[file], retained=retained)["ok"]
    assert _propose(root, files=[file])["ok"]
    owner = _ready(root)
    _forge(root, store_dir, "legacy-keep", [SECOND], [alias])
    _choose(owner, root)
    assert _outcome(owner, root)[2] == "retained_by_other_offer"
    assert (store_dir / exact).read_bytes() == b"one-link evidence retained by the owner"


def test_reserved_spelling_uses_this_directorys_name_lookup(finished):
    root, store_dir, _ = finished
    upper = store_dir / "SUMMARY.JSON"
    upper.write_bytes(b"carrier bytes")
    canonical = store_dir / "summary.json"
    file = {"path": upper.name, "role": "raw", "loss": "exact source"}
    if canonical.exists() and canonical.samefile(upper):
        assert not _propose(root, files=[file])["ok"]
    else:
        canonical.write_bytes(b"separate protected summary")
        assert not canonical.samefile(upper)
        assert _propose(root, files=[file])["ok"]
        _choose(_ready(root), root)
        assert not upper.exists() and canonical.read_bytes() == b"separate protected summary"


def test_native_alias_after_move_obeys_the_same_release_while_a_distinct_sibling_remains_available(finished):
    from ouroboros.gateway.task_archive import serve_task_file
    root, store_dir, _ = finished
    exact, alias = "Case-Snapshot.bin", "case-snapshot.BIN"
    selected, other = store_dir / exact, store_dir / alias
    selected.write_bytes(b"chosen")
    aliased = other.exists() and other.samefile(selected)
    if not aliased:
        other.write_bytes(b"distinct")
    assert _propose(root, files=[{"path": exact, "role": "raw", "loss": "exact bytes"}])["ok"]
    _choose(_ready(root), root)
    assert not selected.exists()
    response = serve_task_file(root, store_dir, alias, alias, None, task_id=TASK)
    if aliased:
        assert response.status_code == 410
    else:
        assert response.status_code == 200 and other.read_bytes() == b"distinct"
        response._handle.close()


@pytest.mark.skipif(os.name != "nt", reason="native Windows names and optional 8.3 aliases")
def test_windows_short_name_obeys_the_release_and_special_selectors_cannot_be_offered(finished):
    from ouroboros.confined_files import short_name
    from ouroboros.gateway.task_archive import serve_task_file

    root, directory, _ = finished
    for name in ("CON.txt", "NUL", "data.bin:stream", "raw.", "raw "):
        assert _propose(root, files=[{"path": name, "role": "raw", "loss": "exact bytes"}])["refused"][0]["reason"] == "bad_path"
    selected = directory / "long synthetic snapshot name.bin"
    selected.write_bytes(b"owner selected")
    alias = short_name(selected)
    if alias.casefold() == selected.name.casefold():
        pytest.skip("this volume did not allocate an 8.3 alias")
    assert (directory / alias).samefile(selected)
    assert not _propose(root, files=[{"path": alias, "role": "raw", "loss": "exact bytes"}])["ok"]
    assert _propose(root, files=[{"path": selected.name, "role": "raw", "loss": "exact bytes"}])["ok"]
    _choose(_ready(root), root)
    assert not selected.exists()
    assert serve_task_file(root, directory, alias, alias, None, task_id=TASK).status_code == 410


def test_a_substituted_directory_between_verification_and_move_cannot_redirect_the_rename(finished, monkeypatch):
    root, directory, _ = finished
    assert _propose(root)["ok"]
    owner = _ready(root)
    move = pr._move_selected
    saved = root.parent / "saved-files"

    def replace_parent(source, target, anchor, stamp):
        source.parent.rename(saved)
        if os.name == "nt":
            import _winapi
            _winapi.CreateJunction(str(saved), str(source.parent))
        else:
            source.parent.symlink_to(saved, target_is_directory=True)
        return move(source, target, anchor, stamp)

    monkeypatch.setattr(pr, "_move_selected", replace_parent)
    _choose(owner, root)
    assert not _record(root)["released"]
    assert (saved / Path(SNAP).name).read_bytes() == b"S" * 70_000
    assert (directory / REPORT).read_bytes() == b"# measured\n"


def test_archive_rechecks_the_spooled_members_before_serving(finished, monkeypatch):
    from ouroboros.gateway import task_archive

    root, directory, _ = finished
    assert _propose(root)["ok"]
    owner = _ready(root)
    read = task_archive.read_release_state
    def confirm_before_response(base, tid):
        _choose(owner, root)
        return read(base, tid)
    monkeypatch.setattr(task_archive, "read_release_state", confirm_before_response)
    response = task_archive.serve_directory_archive(root, TASK, "delegated_runs.zip", "delegated_runs", other_selectors=False)
    assert response.status_code == 503 and json.loads(response.body)["reason_code"] == "artifact_archive_unavailable"
    assert not (directory / SNAP).exists() and (directory / REPORT).is_file()


def test_two_task_ids_share_a_release_only_when_the_directory_is_really_an_alias(finished):
    from ouroboros.gateway.task_archive import serve_task_file

    root, directory, _ = finished
    assert _propose(root)["ok"]
    _choose(_ready(root), root)
    tid = TASK.upper()
    seed_result(root, tid, document(tid, "completed", result="another task"))
    other = release_state.store_of(root, tid)
    aliased = other.exists() and other.samefile(directory)
    replica = other / SNAP
    replica.parent.mkdir(parents=True, exist_ok=True)
    replica.write_bytes(b"synthetic replica")
    response = serve_task_file(root, other, SNAP, replica.name, None, task_id=tid)
    if aliased:
        assert response.status_code == 503 and json.loads(response.body)["reason_code"] == "artifact_unavailable"
    else:
        assert response.status_code == 200
        response._handle.close()
