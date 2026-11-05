"""The mind's Storage -> Free (``free_storage``, owner Q3 = A): the same Free, the same proofs, a receipt and a Main row.

``FreeSpace.request`` runs one preview with the supervisor's probes and then the ordinary pool (or the named
saved groups) through the same retirers the owner's press uses: a finished drive goes, a drive whose task is
live stays with its reason, the owner's Keep holds, a dry run removes nothing. The receipt is the
``storage_free`` job row (request id, actor, preview, outcome), the run's event names the actor, and ``notify``
hears one sentence for Main. A request while another preview or run is under way is refused with a receipt
that says ``busy``; a request id already answered is a duplicate. The tool emits the control event and reads
the receipt back; the supervisor handler is the one that calls ``request``. Isolated roots only.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

from ouroboros import maintenance_jobs, owner_keep, store_import
from ouroboros.storage_free import REQUEST_JOB, FreeSpace
from tests.test_storage_free import OLD, _drives, _fake_sizer, _guard, _join


def _request(owner: FreeSpace, root: Path, repo: Path, **kwargs):
    said = []
    answer = owner.request(root, repo, actor="mind:task-9", request_id=kwargs.pop("request_id", "req-1"),
                           notify=said.append, **kwargs)
    _join(owner)
    if owner._thread is not None:  # the preview's thread handed the run to a second thread
        _join(owner)
    return answer, said


def test_the_minds_request_frees_what_the_owners_press_would_and_keeps_the_rest_with_reasons(tmp_path, monkeypatch):
    root = tmp_path / "data"
    probes = _drives(root, [("done", "direct", "completed", OLD), ("live", "direct", "completed", OLD, True),
                            ("kept", "headless", "completed", OLD)])
    assert owner_keep.set_owner_keep(root, "kept", keep=True, source={"basis": "test"})["keep"] is True
    monkeypatch.setattr("ouroboros.server_maintenance.DATA_DIR", root)
    _fake_sizer(monkeypatch)
    owner = FreeSpace(live=lambda task_id: probes.get(task_id, False), guard=_guard)
    answer, said = _request(owner, root, tmp_path)
    assert answer == {"started": True, "reason": ""}
    assert not (root / "task_drives" / "done").exists()
    assert (root / "task_drives" / "live").is_dir() and (root / "state" / "headless_tasks" / "kept").is_dir()
    receipt = maintenance_jobs.job(root, REQUEST_JOB)
    assert (receipt["request_id"], receipt["actor"], receipt["state"]) == ("req-1", "mind:task-9", "done")
    assert receipt["preview"]["pool"]["count"] == 1, "the live and the kept drive were never offered"
    assert {row["reason"] for row in receipt["preview"]["pool"]["excluded"]} == {"task_live", "owner_keep"}
    assert receipt["outcome"]["removed"] + receipt["outcome"]["deleting"] == 1 and receipt["outcome"]["kept"] == []
    assert len(said) == 1 and said[0].startswith("Free space (mind:task-9): ") and "free now" in said[0]
    assert "removed 1 (" in said[0] or "moved 1 aside (" in said[0], said  # moved into the custody trash: the child deletes it
    rows = [json.loads(line) for line in (root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    event, = [row for row in rows if row["type"] == "storage_free"]
    assert event["report"]["actor"] == "mind:task-9" and event["report"]["removed_ids"] + event["report"]["deleting_ids"] == ["direct:done"]


def test_a_dry_run_removes_nothing_and_a_second_request_waits_its_turn(tmp_path, monkeypatch):
    root = tmp_path / "data"
    probes = _drives(root, [("done", "direct", "completed", OLD)])
    monkeypatch.setattr("ouroboros.server_maintenance.DATA_DIR", root)
    hold = threading.Event()
    _fake_sizer(monkeypatch, hold=hold)
    owner = FreeSpace(live=lambda task_id: probes.get(task_id, False), guard=_guard)
    said = []
    assert owner.request(root, tmp_path, actor="mind:a", request_id="dry-1", dry_run=True, notify=said.append)["started"]
    refused = owner.request(root, tmp_path, actor="mind:b", request_id="busy-1", notify=said.append)
    assert refused == {"started": False, "reason": "busy"}
    busy = maintenance_jobs.job(root, REQUEST_JOB)
    assert (busy["request_id"], busy["state"], busy["refusal"]) == ("busy-1", "failed", "busy"), "the refusal is a receipt too"
    hold.set()
    _join(owner)
    assert (root / "task_drives" / "done").is_dir(), "a dry run removes nothing"
    receipt = maintenance_jobs.job(root, REQUEST_JOB)
    assert (receipt["request_id"], receipt["state"]) == ("dry-1", "done") and "outcome" not in receipt
    assert receipt["preview"]["pool"]["count"] == 1
    assert said == [f"Free space (mind:a): looked only; 1 items could go, {receipt['preview']['pool']['bytes'] / 1e9:.1f} GB."]
    assert owner.request(root, tmp_path, actor="mind:a", request_id="dry-1", notify=said.append) == {"started": False, "reason": "duplicate"}


def test_an_empty_pool_is_a_receipt_that_says_so(tmp_path, monkeypatch):
    root = tmp_path / "data"
    probes = _drives(root, [("live", "direct", "completed", OLD, True)])
    monkeypatch.setattr("ouroboros.server_maintenance.DATA_DIR", root)
    _fake_sizer(monkeypatch)
    owner = FreeSpace(live=lambda task_id: probes.get(task_id, False), guard=_guard)
    answer, said = _request(owner, root, tmp_path, request_id="empty-1")
    assert answer["started"] and (root / "task_drives" / "live").is_dir()
    receipt = maintenance_jobs.job(root, REQUEST_JOB)
    assert receipt["refusal"] == "nothing_selected" and receipt["preview"]["pool"]["count"] == 0
    assert said == ["Free space (mind:task-9): not run, nothing selected."]


def test_the_tool_emits_the_control_event_and_returns_the_receipt_it_finds(tmp_path, monkeypatch):
    from ouroboros.tools import control_storage

    root = tmp_path / "data"
    root.mkdir()
    store_import.ensure_current_store(root)
    emitted = []

    class Queue:
        def put_nowait(self, evt):
            emitted.append(evt)
            maintenance_jobs.record(root, REQUEST_JOB, request_id=evt["request_id"], actor=f"mind:{evt['task_id']}",
                                    state="done", outcome={"removed": 2, "removed_bytes": 3_000_000, "kept": [], "deleting": 0,
                                                           "failed": 0, "free_after": 10 ** 11})

    ctx = SimpleNamespace(task_id="task-9", drive_root=root, budget_drive_root=root, event_queue=Queue(), pending_events=[])
    monkeypatch.setattr("ouroboros.tool_access_paths.canonical_data_root", lambda _ctx: root)
    answer = json.loads(control_storage._free_storage(ctx, groups=["review_checkouts"], wait_sec=5))
    assert emitted[0]["type"] == "free_storage" and emitted[0]["groups"] == ["review_checkouts"] and emitted[0]["task_id"] == "task-9"
    assert answer["ok"] and answer["status"] == "done" and answer["receipt"]["outcome"]["removed"] == 2
    assert answer["request_id"] == emitted[0]["request_id"] and answer["transport"] == "live"
    unknown = json.loads(control_storage._free_storage(ctx, groups=["work_folders"]))
    assert unknown == {"ok": False, "error": "unknown_group", "unknown": ["work_folders"], "groups": list(control_storage.SAVED_GROUPS)}


def test_a_receipt_that_has_not_landed_reads_as_running(tmp_path, monkeypatch):
    from ouroboros.tools import control_storage

    root = tmp_path / "data"
    root.mkdir()
    store_import.ensure_current_store(root)
    ctx = SimpleNamespace(task_id="task-9", drive_root=root, budget_drive_root=root, event_queue=None, pending_events=[])
    monkeypatch.setattr("ouroboros.tool_access_paths.canonical_data_root", lambda _ctx: root)
    answer = json.loads(control_storage._free_storage(ctx, wait_sec=0))
    assert answer["status"] == "queued" and answer["transport"] == "deferred" and ctx.pending_events[0]["type"] == "free_storage"
    assert "storage_free job row" in answer["note"]
