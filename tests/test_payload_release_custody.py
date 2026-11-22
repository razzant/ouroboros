"""Exercise owner Free, child publication, settlement and Keep against synthetic task stores.

These cases use the terminal and closure checks, Keep inventory and stored release receipts.
Failures are injected at the accounting write, rename or publication boundary.
"""

from __future__ import annotations

import errno
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import artifacts, headless, history_retention, owner_keep, task_custody
from ouroboros import payload_release as pr
from ouroboros import payload_release_state as release_state
from ouroboros.gateway.task_owner_keep import api_task_owner_keep
from ouroboros.task_results import write_task_result
from ouroboros.task_store import patch_document, read_document
from supervisor.queue_transitions import task_settlement_interlock
from tests.test_payload_release import (
    CARRIER,
    REPORT,
    SCRIPT,
    SNAP,
    TASK,
    _choose,
    _dual_store,
    _outcome,
    _owner,
    _propose,
    _ready,
    _record,
)
from tests.test_payload_release import real_finished as real_finished  # re-export the real task/closure fixture


@pytest.fixture(autouse=True)
def task_copy_inventory(tmp_path, monkeypatch):
    copies = tmp_path / "copies"
    copies.mkdir()
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(copies))


def _client(root):
    app = Starlette(routes=[Route("/api/tasks/{task_id}/owner-keep", api_task_owner_keep, methods=["POST"])])
    app.state.drive_root = root
    return TestClient(app)


def _keep(client):
    response = client.post(f"/api/tasks/{TASK}/owner-keep", json={"keep": True})
    assert response.status_code == 200, response.text
    return response.json()


def _preview_payload(root, rel, *, also=()):
    assert _propose(root, files=[{"path": path, "role": "synthetic snapshot", "loss": "exact rerun lost"}
                                 for path in (rel, *also)])["ok"]
    return _ready(root, _owner(guard=task_settlement_interlock))


def _protect(canonical):
    return {rel: (canonical / rel).read_bytes() for rel in (REPORT, SCRIPT, CARRIER)}


def _chosen_row(root, chosen):
    return next(row for row in read_document(root, TASK)["artifacts"] if row["path"] == str(chosen))


def _account_fails(monkeypatch):
    write = pr._write

    def fail(root, task_id, change):
        if change.__name__ == "account":
            raise OSError(errno.EIO, "synthetic accounting store failure")
        return write(root, task_id, change)

    monkeypatch.setattr(pr, "_write", fail)


def _copyback(root, drive):
    answer = headless.copy_child_task_result(root, {"id": TASK, "drive_root": str(drive)})
    assert answer is not None
    return headless.retry_child_task_refs(root, drive, TASK)


def _settle(root, drive, *, stop=None):
    return task_custody.settle_child_drive(root, TASK, drive, live=lambda _task: False,
                                         guard=task_settlement_interlock, stop=stop)


def _raw_store(real_finished):
    root, canonical = real_finished
    drive = root / "task_drives" / TASK
    rel = "source_handles/test-evidence/run-large/snapshot.bin"
    source = release_state.store_of(drive, TASK) / rel
    source.parent.mkdir(parents=True)
    source.write_bytes(b"synthetic raw snapshot")
    write_task_result(drive, TASK, "running")
    patch_document(root, TASK, lambda row: {**row, "child_drive_root": str(drive)})
    with task_custody.task_custody_lock(root, TASK) as locked:
        assert locked
        state = {"pending_refs": []}
        assert history_retention.carry_raw_evidence(root, drive, TASK, state) == 1 and not state["pending_refs"]
    return root, canonical, drive, source, rel


@pytest.mark.parametrize("publisher", ["copyback", "raw", "settlement"])
@pytest.mark.parametrize("order", ["publish_first", "release_first"])
def test_pending_move_blocks_real_publishers_until_recovery(real_finished, monkeypatch, publisher, order):
    if publisher == "raw":
        root, canonical, drive, source, rel = _raw_store(real_finished)
    else:
        root, canonical, drive, child, saved = _dual_store(real_finished)
        source, rel = Path(child["path"]), saved["name"]
    originals, source_bytes = _protect(canonical), source.read_bytes()

    def publish():
        if publisher == "copyback":
            return _copyback(root, drive)["child_ref_promotion"]["pending_refs"]
        if publisher == "raw":
            state = {"pending_refs": []}
            with task_custody.task_custody_lock(root, TASK) as locked:
                assert locked
                history_retention.carry_raw_evidence(root, drive, TASK, state)
            return state["pending_refs"]
        # Exercise real Phase A but retain the source for the reverse-order replay (before Phase B).
        ready = None
        if order == "publish_first":
            before = task_custody.custody_revision(read_document(root, TASK))

            def ready():
                return task_custody.custody_revision(read_document(root, TASK)) != before
        answer = _settle(root, drive, stop=ready)
        return [{"reason": answer["reason"]}] if answer["status"] == "retained" else []

    if order == "publish_first":
        if publisher == "settlement":
            # Phase A leaves a source-row rewrite; the generation fence stops before the drive move.
            patch_document(root, TASK, lambda row: {**row, "artifacts": [child]})
        early = publish()
        assert not early or publisher == "settlement" and early == [{"reason": "generation_closed"}]
        assert (canonical / rel).read_bytes() == source_bytes and source.exists()
    owner = _preview_payload(root, rel)
    with monkeypatch.context() as failure:
        _account_fails(failure)
        assert _choose(owner, root)["started"]
    assert _outcome(owner, root)[2] == "release_pending"
    assert rel in release_state.pending_files(read_document(root, TASK)) and rel not in _record(root)["released"]
    assert not (canonical / rel).exists()
    gaps = publish()  # replay is held even when this publisher ran before the owner's decision
    assert any("payload_release_pending" in gap["reason"] for gap in gaps), gaps
    assert not (canonical / rel).exists() and source.read_bytes() == source_bytes
    assert _protect(canonical) == originals
    pr.recover(root)
    pr.recover(root)  # idempotent: no second move or decision
    assert not release_state.pending_files(read_document(root, TASK)) and rel in _record(root)["released"]
    assert _outcome(owner, root)[0] == "deleting" and len(_record(root)["operations"]) == 1
    assert not (canonical / rel).exists() and _protect(canonical) == originals


def _collision_store(real_finished, *, first_publisher="copyback"):
    root, canonical = real_finished
    drive = root / "task_drives" / TASK
    material = root.parent / "snapshot.bin"
    original, selected = b"different original payload", b"chosen immutable payload"

    def capture(data_root, content, name):
        material.write_bytes(content)
        identity = {"name": name, "sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
        return artifacts.copy_file_to_task_artifacts(SimpleNamespace(drive_root=data_root, task_id=TASK), material,
                                                    immutable=True, expected=identity)

    canonical_row = capture(root, original, "snapshot.bin")
    child = capture(drive, selected, "snapshot.bin")
    other = capture(drive, selected, "independent.bin")  # equal SHA alone never releases this address
    write_task_result(drive, TASK, "completed" if first_publisher == "copyback" else "running",
                      result="synthetic measurement complete", artifacts=[child, other])
    patch_document(root, TASK, lambda row: {**row, "child_drive_root": str(drive), "artifacts": [canonical_row]})
    if first_publisher == "copyback":
        copied = _copyback(root, drive)
        saved = next(row for row in copied["artifacts"] if Path(row["path"]).name.startswith("snapshot_"))
    else:
        before = task_custody.custody_revision(read_document(root, TASK))
        outcome = _settle(root, drive, stop=lambda: task_custody.custody_revision(read_document(root, TASK)) != before)
        assert outcome["status"] == "retained" and outcome["reason"] == "generation_closed", outcome
        copied = read_document(root, TASK)
        saved = next(row for row in copied["artifacts"] if Path(row["path"]).name.startswith("snapshot.")
                     and Path(row["path"]) != canonical / "snapshot.bin")
    chosen = Path(saved["path"])
    assert chosen.read_bytes() == selected and (canonical / "snapshot.bin").read_bytes() == original
    return root, canonical, drive, child, chosen, original, selected


@pytest.mark.parametrize("publisher", ["copyback", "settlement"])
@pytest.mark.parametrize("proof", ["same", "changed", "unproved"])
@pytest.mark.parametrize("release_original", [False, True])
def test_copyback_collision_release_is_not_revived_under_either_name(real_finished, publisher, proof, release_original):
    root, canonical, drive, child, chosen, original, selected = _collision_store(real_finished)
    protected, chosen_row = _protect(canonical), _chosen_row(root, chosen)
    _choose(_preview_payload(root, chosen.name, also=("snapshot.bin",) if release_original else ()), root)
    if proof == "changed":
        Path(child["path"]).chmod(0o644)
        Path(child["path"]).write_bytes(b"new and unchosen bytes")
    elif proof == "unproved":
        def forget_digest(row):
            row[release_state.KEY]["released"][chosen.name].pop("sha256")
            return row
        patch_document(root, TASK, forget_digest)
    remaining = Path(child["path"]).read_bytes()
    if publisher == "copyback":
        answer = _copyback(root, drive)
        gaps = answer["child_ref_promotion"]["pending_refs"]
        assert bool(gaps) == (proof != "same"), gaps
    else:
        answer = _settle(root, drive)
        assert answer["status"] == ("removed" if proof == "same" else "retained"), answer
    if proof != "same":
        assert Path(child["path"]).read_bytes() == remaining
    else:
        assert _chosen_row(root, chosen) == chosen_row
    assert not chosen.exists() and not task_custody._collision(canonical / "snapshot.bin", child["sha256"]).exists()
    if release_original:
        assert not (canonical / "snapshot.bin").exists()
    else:
        assert (canonical / "snapshot.bin").read_bytes() == original
    assert (canonical / "independent.bin").read_bytes() == selected
    assert _protect(canonical) == protected


@pytest.mark.parametrize("release_original", [False, True])
def test_settlement_collision_released_before_copyback_stays_released(real_finished, release_original):
    root, canonical, drive, _child, chosen, original, selected = _collision_store(real_finished, first_publisher="settlement")
    chosen_row = _chosen_row(root, chosen)
    _choose(_preview_payload(root, chosen.name, also=("snapshot.bin",) if release_original else ()), root)
    answer = _copyback(root, drive)
    assert not answer["child_ref_promotion"]["pending_refs"]
    assert _chosen_row(root, chosen) == chosen_row
    assert not chosen.exists() and not list(canonical.glob("snapshot_*.bin"))
    if release_original:
        assert not (canonical / "snapshot.bin").exists()
    else:
        assert (canonical / "snapshot.bin").read_bytes() == original
    assert (canonical / "independent.bin").read_bytes() == selected


@pytest.mark.parametrize("publisher", ["copyback", "settlement"])
def test_pending_copyback_collision_holds_both_publishers(real_finished, monkeypatch, publisher):
    root, canonical, drive, child, chosen, original, _selected = _collision_store(real_finished)
    owner = _preview_payload(root, chosen.name)
    with monkeypatch.context() as failure:
        _account_fails(failure)
        _choose(owner, root)
    if publisher == "copyback":
        gaps = _copyback(root, drive)["child_ref_promotion"]["pending_refs"]
        assert any("payload_release_pending" in row["reason"] for row in gaps), gaps
    else:
        answer = _settle(root, drive)
        assert answer["status"] == "retained" and answer["reason"] == "payload_release_pending", answer
    assert Path(child["path"]).exists() and not chosen.exists()
    assert not task_custody._collision(canonical / "snapshot.bin", child["sha256"]).exists()
    assert (canonical / "snapshot.bin").read_bytes() == original
    assert not _record(root)["released"]


def test_copyback_keeps_every_pending_address_after_first_failed_row(real_finished, monkeypatch):
    root, canonical, drive, _child, chosen, _original, _selected = _collision_store(real_finished)
    paths = [chosen.name, "independent.bin"]
    assert _propose(root, files=[{"path": path, "role": "synthetic snapshot", "loss": "exact rerun lost"}
                                 for path in paths])["ok"]
    owner = _ready(root, _owner(guard=task_settlement_interlock))
    with monkeypatch.context() as failure:
        _account_fails(failure)
        _choose(owner, root)
    answer = _copyback(root, drive)
    assert all(not (canonical / path).exists() for path in paths)
    assert set(paths) <= release_state.pending_files(read_document(root, TASK))
    assert len(answer["child_ref_promotion"]["pending_refs"]) >= 2
    assert not _record(root)["released"]


def test_result_material_retain_does_not_publish_over_a_pending_move(real_finished, monkeypatch):
    from ouroboros.result_material import _retained_copy

    root, canonical = real_finished
    source = root.parent / "material.bin"
    source.write_bytes(b"synthetic result material")
    item = {"path": str(source), "size": source.stat().st_size, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    target = Path(_retained_copy(root, TASK, item, {})["retained"])
    owner = _preview_payload(root, target.relative_to(canonical).as_posix())
    with monkeypatch.context() as failure:
        _account_fails(failure)
        _choose(owner, root)
    assert not target.exists() and not _record(root)["released"]
    assert "pending accounting" in _retained_copy(root, TASK, item, {})["retained_gap"]
    assert not target.exists() and source.read_bytes() == b"synthetic result material"
    pr.recover(root)
    assert "released by the owner" in _retained_copy(root, TASK, item, {})["retained_gap"]


def _retire_empty_drive(real_finished):
    root, canonical = real_finished
    drive = root / "task_drives" / TASK
    write_task_result(drive, TASK, "running")
    patch_document(root, TASK, lambda row: {**row, "child_drive_root": str(drive)})
    outcome = _settle(root, drive)
    assert outcome["status"] == "removed" and not drive.exists(), outcome
    return root, canonical


@pytest.mark.parametrize("offered", [False, True])
def test_keep_endpoint_keeps_last_canonical_material_after_drive_retirement(real_finished, offered):
    root, canonical = _retire_empty_drive(real_finished)
    if offered:
        assert _propose(root)["ok"]
    with _client(root) as client:
        answer = _keep(client)
    assert answer["keep"] and answer["status"] == "partly_retired", answer
    assert {str(canonical / rel) for rel in (SNAP, REPORT, SCRIPT, CARRIER)} <= set(answer["retained"])
    fact = read_document(root, TASK)["owner_keep"]
    assert fact["source"]["basis"] == "authenticated_gateway_request"
    assert pr.holds(root, TASK, read_document(root, TASK), lambda _task: False) == "owner_keep"


@pytest.mark.parametrize("failure", ["unreadable", "malformed", "accounted_shape"])
def test_keep_saves_fact_when_payload_read_fails(real_finished, monkeypatch, failure):
    """From f6bfcc318, in this branch's per-file format: a canonical store that cannot be read (OSError) or a release
    record that is malformed (ValueError from ``release_record``) still saves an explicit Keep; the listed store files
    are answered ``unknown`` (``canonical_payloads``), never retained without a state. A release (keep=False) on the
    same failure raises and leaves the saved fact in place."""
    root, canonical = _retire_empty_drive(real_finished)
    assert _propose(root)["ok"]
    if failure == "unreadable":
        def unreadable_store(*_args):
            raise PermissionError(errno.EACCES, "synthetic canonical inventory unavailable")

        monkeypatch.setattr(release_state, "keep_places", unreadable_store)
        expected = PermissionError
    elif failure == "malformed":
        patch_document(root, TASK, lambda row: {**row, "payload_release": {"released": []}})
        expected = ValueError
    else:  # an accounted operation whose files are no list: the reader answers its own ValueError
        patch_document(root, TASK, lambda row: {**row, "payload_release": {
            "operations": {"old": {"state": "accounted", "files": 1}}}})
        expected = ValueError
    with _client(root) as client:
        answer = _keep(client)
    assert answer["keep"] and answer["status"] == "partly_retired", answer
    assert "canonical_payloads" in answer["unknown"]
    for rel in (SNAP, REPORT, SCRIPT, CARRIER):
        path = str(canonical / rel)
        assert answer["releases"].get(path) == "unknown", answer
        assert path not in answer["retained"] and path not in answer["retired"], answer
    fact = read_document(root, TASK)["owner_keep"]
    assert fact["source"]["basis"] == "authenticated_gateway_request"
    assert pr.holds(root, TASK, read_document(root, TASK), lambda _task: False) == "owner_keep"
    with pytest.raises(expected):
        owner_keep.set_owner_keep(root, TASK, keep=False, source={"basis": "test"})
    assert read_document(root, TASK)["owner_keep"] == fact


@pytest.mark.parametrize("order", ["keep_first", "free_first"])
def test_real_keep_and_free_in_both_orders(real_finished, order):
    root, canonical = _retire_empty_drive(real_finished)
    protected = _protect(canonical)
    owner = _preview_payload(root, SNAP)
    with _client(root) as client:
        if order == "keep_first":
            assert _keep(client)["keep"]
        _choose(owner, root)
        answer = _keep(client)
        if order == "keep_first":
            assert _outcome(owner, root)[2] == "owner_keep" and (canonical / SNAP).exists()
            assert str(canonical / SNAP) in answer["retained"]
        else:
            assert not (canonical / SNAP).exists() and str(canonical / SNAP) in answer["retired"]
            assert answer["releases"][str(canonical / SNAP)] == "deleting"
            task_custody.remove_custody_trash(root)
            assert _keep(client)["releases"][str(canonical / SNAP)] == "deleted"
    assert answer["keep"] and _protect(canonical) == protected


@pytest.mark.parametrize("stage", ["queued", "moved", "staged", "deleting"])
def test_keep_endpoint_distinguishes_decision_move_accounting_and_trash(real_finished, monkeypatch, stage):
    root, canonical = _retire_empty_drive(real_finished)
    protected = _protect(canonical)
    owner = _preview_payload(root, SNAP)
    with monkeypatch.context() as failure:
        if stage in {"queued", "moved"}:
            _account_fails(failure)
        if stage == "queued":
            def fail_rename(*_args):
                raise OSError(errno.EIO, "synthetic rename failure")
            failure.setattr(pr, "_move_selected", fail_rename)
        elif stage == "staged":
            def fail_publish(*_args):
                raise OSError(errno.EIO, "synthetic trash handoff failure")
            failure.setattr(pr, "_publish", fail_publish)
        _choose(owner, root)
    with _client(root) as client:
        answer = _keep(client)
    assert answer["keep"] and answer["releases"][str(canonical / SNAP)] == stage, answer
    assert (str(canonical / SNAP) in answer["retained"]) == (stage == "queued")
    assert (str(canonical / SNAP) in answer["retired"]) == (stage != "queued")
    if stage in {"queued", "moved"}:
        assert not _record(root)["released"], "moving is not a receipt of deletion"
        pr.recover(root)  # under Keep: account the past, never move the source or hand staging to trash
        assert not release_state.pending_files(read_document(root, TASK))
        assert (canonical / SNAP).exists() == (stage == "queued")
        assert not list((root / "state" / "custody_trash").glob(f"*{pr.MARK}*"))
        with _client(root) as client:
            assert _keep(client)["releases"][str(canonical / SNAP)] == ("held" if stage == "queued" else "staged")
    assert {str(canonical / rel) for rel in (REPORT, SCRIPT, CARRIER)} <= set(answer["retained"])
    assert _protect(canonical) == protected


def test_keep_rechecks_payloads_when_free_wins_after_inventory(real_finished, monkeypatch):
    root, canonical = _retire_empty_drive(real_finished)
    owner = _preview_payload(root, SNAP)
    inventory = owner_keep.inventory

    def release_after_read(*args):
        found = inventory(*args)
        _choose(owner, root)
        return found

    monkeypatch.setattr(owner_keep, "inventory", release_after_read)
    with _client(root) as client:
        answer = _keep(client)
    assert answer["keep"] and str(canonical / SNAP) in answer["retired"]
    assert str(canonical / SNAP) not in answer["retained"]
    assert answer["releases"][str(canonical / SNAP)] == "deleting"


@pytest.mark.parametrize("order", ["keep_first", "recover_first"])
def test_keep_does_not_call_unexplained_staging_deleted(real_finished, monkeypatch, order):
    root, canonical = _retire_empty_drive(real_finished)
    protected = _protect(canonical)
    owner = _preview_payload(root, SNAP)
    with monkeypatch.context() as failure:
        _account_fails(failure)
        _choose(owner, root)
    [operation] = _record(root)["operations"].values()
    staging = root / "state" / "custody_staging" / operation["staging"] / "released"
    slot = staging / "0"
    original = slot.read_bytes()
    slot.rename(staging / "unexplained")  # simulate an outside writer; neither this name nor replacement is chosen
    slot.write_bytes(b"unselected replacement")
    with _client(root) as client:
        if order == "keep_first":
            assert _keep(client)["releases"][str(canonical / SNAP)] == "unknown"
        assert pr.recover(root)["held"] == [staging.parent.name]
        answer = _keep(client)
    assert answer["keep"] and "canonical_payloads" in answer["unknown"]
    assert answer["releases"][str(canonical / SNAP)] == "unknown"
    assert str(canonical / SNAP) not in answer["retired"]
    assert not _record(root)["released"]
    assert (staging / "unexplained").read_bytes() == original and slot.read_bytes() == b"unselected replacement"
    assert not list((root / "state" / "custody_trash").glob(f"*{pr.MARK}*"))
    assert _protect(canonical) == protected


@pytest.mark.parametrize("publisher", ["copyback", "settlement"])
def test_selected_collision_keeps_exact_disposition_and_original_download(real_finished, publisher):
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools.core import _read_file
    from ouroboros.tools.join_ledger import _child_result_sha256, _current_child_result_disposition
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.task_tree import _tree_note
    from tests.test_child_result_disposition import _parent_ctx, _payload
    from tests.test_task_file_serving import _client as file_client

    root, canonical, drive, child, chosen, original, selected = _collision_store(real_finished, first_publisher=publisher)
    parent = "parent1"
    write_task_result(root, parent, "completed", result="used the measurement", reason_code="final_message",
                      root_task_id=parent, delegation_role="root")
    patch_document(root, TASK, lambda row: {**row, "parent_task_id": parent, "root_task_id": parent, "delegation_role": "subagent"})
    def current():
        return load_effective_task_result(root, TASK)

    before = current()
    digest = _child_result_sha256(before)
    reader = ToolContext(repo_dir=root.parent, drive_root=root, task_id=TASK)
    assert selected.decode() in _read_file(reader, chosen.name, root="artifact_store")
    logical = _chosen_row(root, chosen)["name"]
    assert logical == ("snapshot.bin" if publisher == "copyback" else chosen.name)
    decision = _tree_note(_parent_ctx(root), "decision", "accepted the measured artifact", payload=_payload(TASK, "integrated", digest))
    assert decision.startswith("OK:"), decision
    assert _current_child_result_disposition(current()) == "integrated"
    _choose(_preview_payload(root, chosen.name), root)
    after = current()
    assert _child_result_sha256(after) == digest
    assert _current_child_result_disposition(after) == "integrated"
    [fact] = release_state.notice(after)["files"]
    assert fact["path"] == chosen.name and fact["bytes"] == len(selected) and fact["bytes_now"] == "deleting"
    assert not chosen.exists() and (canonical / "snapshot.bin").read_bytes() == original
    assert "artifact_released" in _read_file(reader, chosen.name, root="artifact_store")
    assert Path(child["path"]).read_bytes() == selected
    from ouroboros.gateway.task_archive import serve_task_file
    assert serve_task_file(root, release_state.store_of(drive, TASK), "snapshot.bin", "snapshot.bin", child, task_id=TASK).status_code == 410
    with pytest.raises(release_state.ReleaseUnavailable, match="artifact_released"):
        artifacts.copy_file_to_task_artifacts(SimpleNamespace(drive_root=root, task_id=parent), child["path"],
                                             immutable=True, expected=child)
    with pytest.raises(release_state.ReleaseUnavailable, match="artifact_released"):
        artifacts.store_task_artifact_bytes(root, TASK, chosen.name, selected)
    assert not chosen.exists()
    with file_client(root) as client:
        base = f"/api/tasks/{TASK}/artifacts/"
        assert client.get(base + chosen.name, params={"relpath": chosen.name}).status_code == 410
        assert client.get(base + "snapshot.bin").content == original
        assert client.get(base + "independent.bin").content == selected
    # The completed child supplies the effective answer during copyback. A still-running child in the
    # settlement case does not override the canonical terminal answer (effective_task_result).
    answer_root = drive if publisher == "copyback" else root
    patch_document(answer_root, TASK, lambda row: {**row, "result": "new measurement result"})
    changed = current()
    assert changed["result"] == "new measurement result"
    assert _child_result_sha256(changed) != digest and _current_child_result_disposition(changed) != "integrated"


@pytest.mark.parametrize("kind", ["result_material", "raw_evidence"])
def test_unlisted_source_keeps_hash_membership_and_reports_physical_release(real_finished, kind):
    from ouroboros.result_material import _retained_copy
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools.join_ledger import _child_result_sha256
    from tests.test_task_file_serving import _client as file_client

    root, canonical = real_finished
    if kind == "result_material":
        source = root.parent / "observation.bin"
        source.write_bytes(b"synthetic observation")
        item = {"path": str(source), **artifacts.stream_artifact_file(source)}
        path = Path(_retained_copy(root, TASK, item, None)["retained"])
        rel = path.relative_to(canonical).as_posix()
    else:
        root, canonical, drive, source, rel = _raw_store(real_finished)
        path = canonical / rel
    expected = path.read_bytes()
    def absence(document):
        for rows in (document["artifacts"], document["artifact_bundle"]["artifacts"]):
            assert all(row.get("path") != str(path) and row.get("relpath") != rel for row in rows)

    before = load_effective_task_result(root, TASK)
    absence(before)
    _choose(_preview_payload(root, rel), root)
    after = load_effective_task_result(root, TASK)
    absence(after)
    assert _child_result_sha256(before) == _child_result_sha256(after)
    [receipt] = release_state.notice(after)["files"]
    assert (receipt["path"], receipt["bytes"], receipt["bytes_now"]) == (rel, len(expected), "deleting")
    assert not path.exists()
    if kind == "result_material":
        assert "released" in _retained_copy(root, TASK, item, None)["retained_gap"]
        ref = artifacts.actor_source_reference(category="tool_results", source_id="result-material-" + item["sha256"][:16],
                                               data=expected, extension="bin")
        assert ref["path"] == rel
        with pytest.raises(release_state.ReleaseUnavailable, match="artifact_released"):
            artifacts.store_actor_source_bytes(root, TASK, category="tool_results",
                                             source_id="result-material-" + item["sha256"][:16], data=expected, extension="bin")
        path.write_bytes(expected)
        with pytest.raises(release_state.ReleaseUnavailable, match="artifact_released"):
            artifacts.read_actor_source_bytes(root, TASK, ref, anchor=root)
    else:
        assert source.read_bytes() == expected
    with file_client(root) as client:
        for selector in ("source", "relpath"):
            response = client.get(f"/api/tasks/{TASK}/artifacts/{path.name}", params={selector: rel})
            assert response.status_code == 410 and response.json()["reason_code"] == "artifact_released"


def test_registered_download_reads_one_address_and_keeps_busy_pending_and_released_distinct(real_finished, monkeypatch):
    from ouroboros import task_store
    from ouroboros.store import StoreBusy
    from ouroboros.tools.core_file_tools import read_admission
    from ouroboros.tools.registry import ToolContext
    from tests.test_task_file_serving import _client as file_client

    root, canonical, _drive, _child, row = _dual_store(real_finished)
    path = Path(row["path"])
    reader = ToolContext(repo_dir=root.parent, drive_root=root, task_id=TASK)

    def assert_native_refusal(reason):
        _, _, refusal = read_admission(reader, "artifact_store", path.name)
        assert refusal is not None
        assert (refusal.status, refusal.code, refusal.meta["reason_code"]) == (
            "unavailable", "TOOL_ADMISSION_UNAVAILABLE", reason)
        assert reason in refusal.text

    original = path.read_bytes()
    owner = _preview_payload(root, path.name)
    read = task_store.read_document
    calls = []
    def counted(base, tid, **kwargs):
        calls.append((base, tid))
        return read(base, tid, **kwargs)
    def busy(*args, **kwargs):
        raise StoreBusy("synthetic contention")

    with file_client(root) as client:
        url = f"/api/tasks/{TASK}/artifacts/{path.name}"
        with monkeypatch.context() as observing:
            observing.setattr(task_store, "read_document", counted)
            assert client.get(url).content == original
            assert calls == [(root, TASK)]
            observing.setattr(task_store, "read_document", busy)
            response = client.get(url)
            assert response.status_code == 503 and response.json()["cause"] == "StoreBusy"
            assert_native_refusal("artifact_unavailable")
        with monkeypatch.context() as failure:
            _account_fails(failure)
            _choose(owner, root)
        path.write_bytes(original)  # a stale replica reappears while accounting is owed
        proposed = _propose(root, files=[{"path": path.name, "role": "raw", "loss": "the original run"}])
        assert proposed.get("error") == "RELEASE_PENDING"
        response = client.get(url)
        assert response.status_code == 503 and response.json()["reason_code"] == "payload_release_pending"
        assert_native_refusal("payload_release_pending")
        pr.recover(root)
        patch_document(root, TASK, lambda row: {**row, "status": "running"})
        response = client.get(url)
        assert response.status_code == 410 and response.json()["reason_code"] == "artifact_released"
        assert_native_refusal("artifact_released")


def test_publisher_resolves_canonical_rows_only_once_for_matching_receipts(real_finished, monkeypatch):
    root, canonical, drive, child, chosen, _original, _selected = _collision_store(real_finished)
    scans = []
    lookup = release_state.PublicationReleases.canonical_rows.func
    def observe(state):
        scans.append(state)
        return lookup(state)
    from functools import cached_property
    measured = cached_property(observe)
    measured.__set_name__(release_state.PublicationReleases, "canonical_rows")
    monkeypatch.setattr(release_state.PublicationReleases, "canonical_rows", measured)
    _copyback(root, drive)
    assert not scans
    _choose(_preview_payload(root, chosen.name), root)
    state = release_state.PublicationReleases(read_document(root, TASK), canonical)
    src = Path(child["path"])
    assert not state.collision(src.with_name("independent.bin"), canonical / "independent.bin", child)
    with monkeypatch.context() as untouched, src.with_name("independent.bin").open("rb") as handle:
        untouched.setattr(artifacts, "stream_artifact_file", lambda *a, **k: pytest.fail("unrelated download hashed"))
        assert state.download_address(src.parent, "independent.bin", None, handle=handle) == "independent.bin"
    assert not scans
    for _ in range(2):
        assert state.collision(src, canonical / src.name, child)["path"] == str(chosen)
    assert scans == [state]


def _child_payload_preview(real_finished, monkeypatch):
    """Preview a finished child's exact file with a closed root and the supervisor's real interlock."""
    from ouroboros.storage_free import FreeSpace
    from tests._budget_pause_exact_helpers import _install_queue

    root, canonical = real_finished
    parent = "sample-parent"
    _install_queue(root, monkeypatch)
    write_task_result(root, parent, "completed", result="Sample study complete", chat_id=7,
                      origin_message_text="Study the sample rows.",
                      origin_message_ref={"chat_id": 7, "client_message_id": "study-request"},
                      root_task_id=parent, delegation_role="root", reason_code="final_message")
    patch_document(root, TASK, lambda row: {**row, "root_task_id": parent, "parent_task_id": parent,
                                            "delegation_role": "subagent"})
    assert _propose(root)["ok"]
    owner = _ready(root, FreeSpace(offered=("stored_payloads",)))
    [choice] = owner.items(root, owner.projection(root)["plan"], "manual", 0, 10, None)["items"]
    assert choice["task_id"] == TASK
    return root, canonical, parent, owner, [choice["id"]]


@pytest.mark.parametrize("order", ["continue-first", "continue-at-barrier", "release-first"])
def test_root_continue_and_child_saved_file_release(real_finished, monkeypatch, order):
    from threading import Event

    from supervisor.continuation_admission import admit_continuation
    from tests.test_continue_author_stop import NONCE, STOP_AXES
    from tests.test_storage_free import _join

    root, canonical, parent, owner, choice = _child_payload_preview(real_finished, monkeypatch)
    kept = _protect(canonical)
    original = (canonical / SNAP).read_bytes()
    answer = read_document(root, TASK)["result"]

    def continue_parent():
        # The terminal row gains its author's stop decision after preview. Continue now has an eligible root.
        patch_document(root, parent, lambda row: {**row, "outcome_axes": STOP_AXES})
        admitted = admit_continuation(parent, action_nonce=NONCE)
        assert admitted["ok"], admitted
        assert read_document(root, parent)["continued_by"]["action_nonce"] == NONCE

    if order == "continue-first":
        continue_parent()
    if order == "continue-at-barrier":
        prepared, resume = Event(), Event()

        def interlock(stop=None):
            prepared.set()  # Free has prepared its reads, but has not acquired Q.
            assert resume.wait(30), "the test did not release the interlock rendezvous"
            return task_settlement_interlock(stop=stop)

        monkeypatch.setattr(owner, "_guard", interlock)
        try:
            started = owner.start(root, root.parent, owner.projection(root)["plan"], None, None,
                                  manual=choice, accept_loss=True)
            assert started["started"] and prepared.wait(30), started
            continue_parent()  # The real admission commits in the other thread.
        finally:
            resume.set()
            _join(owner)
    else:
        assert _choose(owner, root, choice)["started"]
    if order != "release-first":
        cause = "continuable" if order == "continue-first" else "task_not_proven_over"
        assert _outcome(owner, root)[1:] == ("in_use", cause)
        assert (canonical / SNAP).read_bytes() == original
        assert not _record(root)["operations"]
    else:
        assert not (canonical / SNAP).exists()
        continue_parent()
        assert not (canonical / SNAP).exists()
        assert SNAP in _record(root)["released"]
    assert _protect(canonical) == kept and read_document(root, TASK)["result"] == answer


@pytest.mark.parametrize("keep_child", [True, False], ids=["child-keep", "parent-keep"])
def test_keep_after_preview_applies_to_exactly_one_tasks_files(real_finished, monkeypatch, keep_child):
    root, canonical, parent, owner, choice = _child_payload_preview(real_finished, monkeypatch)
    # Give the parent's Keep a surviving material of its own.
    parent_report = release_state.store_of(root, parent) / "report.md"
    parent_report.parent.mkdir(parents=True, exist_ok=True)
    parent_report.write_bytes(b"parent report")
    kept = _protect(canonical)
    with _client(root) as client:
        selected = TASK if keep_child else parent
        response = client.post(f"/api/tasks/{selected}/owner-keep", json={"keep": True})
        assert response.status_code == 200 and response.json()["keep"]
    assert _choose(owner, root, choice)["started"]
    if keep_child:
        assert _outcome(owner, root)[2] == "owner_keep"
        assert (canonical / SNAP).is_file() and not _record(root)["operations"]
    else:
        assert SNAP in _record(root)["released"] and not (canonical / SNAP).exists()
    assert parent_report.read_bytes() == b"parent report" and _protect(canonical) == kept


def _observe_custody(monkeypatch):
    """Which custody locks are held, with their wait, seen through the public lock its callers import at call time."""
    import contextlib

    lock, held = task_custody.task_custody_lock, []

    @contextlib.contextmanager
    def observed(drive_root, task_id, **kwargs):
        with lock(drive_root, task_id, **kwargs) as locked:
            entry = (task_id, kwargs.get("timeout_sec"))
            if locked:
                held.append(entry)
            try:
                yield locked
            finally:
                if locked:
                    held.remove(entry)

    monkeypatch.setattr(task_custody, "task_custody_lock", observed)
    return held


def test_an_approved_child_release_moves_under_its_own_and_its_roots_custody(real_finished, monkeypatch):
    root, canonical, parent, owner, choice = _child_payload_preview(real_finished, monkeypatch)
    held, at_move, move, kept = _observe_custody(monkeypatch), [], pr._move_selected, _protect(canonical)
    monkeypatch.setattr(pr, "_move_selected", lambda *args: at_move.append(set(held)) or move(*args))
    assert _choose(owner, root, choice)["started"]
    assert at_move and {(TASK, task_custody._INTERLOCK_LOCK_TIMEOUT_SEC), (parent, 0)} <= at_move[0]
    assert SNAP in _record(root)["released"] and not (canonical / SNAP).exists() and _protect(canonical) == kept


def test_a_changed_line_read_at_acceptance_keeps_the_exact_selected_bytes(real_finished, monkeypatch):
    from supervisor.task_ownership import CHANGED, TaskOwnershipRead

    root, canonical, parent, owner, choice = _child_payload_preview(real_finished, monkeypatch)
    held, recheck = _observe_custody(monkeypatch), TaskOwnershipRead.recheck
    kept, original = _protect(canonical), (canonical / SNAP).read_bytes()
    # Only the final recheck runs while the root's custody is held without waiting.
    monkeypatch.setattr(TaskOwnershipRead, "recheck",
                        lambda reads, task_ids: CHANGED if (parent, 0) in held else recheck(reads, task_ids))
    assert _choose(owner, root, choice)["started"]
    assert _outcome(owner, root)[1:] == ("changed", "changed")
    assert (canonical / SNAP).read_bytes() == original and not _record(root)["operations"]
    assert [offer["state"] for offer in _record(root)["offers"].values()] == ["open"]
    release_state.require_file_available(canonical / SNAP)  # the refusal left ordinary reads as they were
    assert _protect(canonical) == kept


def test_recovery_publishes_only_decided_slots_and_keeps_an_unknown_sibling(real_finished, monkeypatch):
    root, canonical = _retire_empty_drive(real_finished)
    owner = _preview_payload(root, SNAP)
    with monkeypatch.context() as failure:
        _account_fails(failure)
        _choose(owner, root)
    [(op, operation)] = _record(root)["operations"].items()
    staging = root / "state" / "custody_staging" / operation["staging"]
    (staging / "sibling.txt").write_bytes(b"not chosen")
    assert pr.recover(root)["held"] == [staging.name]
    pr.recover(root)  # idempotent: nothing more moves, the sibling stays
    trash = root / "state" / "custody_trash" / staging.name
    assert [p.name for p in trash.iterdir()] == ["released"] and [p.name for p in (trash / "released").iterdir()] == ["0"]
    assert (staging / "sibling.txt").read_bytes() == b"not chosen" and not (staging / "released").exists()
    accounted = _record(root)["operations"][op]
    assert accounted["moved"] == [SNAP]
    assert {"entry": "sibling.txt", "reason": "staging_unknown", "bytes": 10} in accounted["staging_held"]
    task_custody.remove_custody_trash(root)
    assert not trash.exists() and (staging / "sibling.txt").read_bytes() == b"not chosen"
