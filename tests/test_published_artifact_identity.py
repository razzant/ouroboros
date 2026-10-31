"""A ready host artifact is published with its measured byte identity (#1969).

The real chain: the child finishes → copy-back exposes the early answer → terminal
file preparation publishes the host artifacts → the parent reads the exact result
hash and records a disposition on it → ordinary deferred history retention and the
child-drive custody pass. Retaining the SAME bytes keeps that exact decision; changed
bytes, a new artifact, a late host notice or a read error still reopen it.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import artifacts, headless, store, task_custody
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.task_status import load_effective_task_result
from ouroboros.tools.control_task_results import _get_task_result
from ouroboros.tools.join_ledger import _child_result_sha256, _current_child_result_disposition
from ouroboros.tools.task_tree import _tree_note

PARENT, CHILD = "parent1", "child1"
HOST_KINDS = {"workspace_patch_manifest", "memory_export", "deliverable_manifest"}


def _ctx(data: Path) -> SimpleNamespace:
    return SimpleNamespace(drive_root=str(data), budget_drive_root=str(data), task_id=PARENT, role="orchestrator",
                           task_metadata={"budget_drive_root": str(data), "root_task_id": PARENT})


def _shown(ctx: SimpleNamespace, data: Path):
    """The parent's real read; its printed hash is the one a disposition binds."""
    text = _get_task_result(ctx, CHILD)
    match = re.search(r"^child_result_sha256=([0-9a-f]{64})$", text, re.M)
    effective = load_effective_task_result(data, CHILD)
    assert match and match[1] == _child_result_sha256(effective), text
    return effective, match[1], text


def _bytes_on_disk(result: dict) -> dict:
    """Independent measurement: what the files hold, whatever the rows claim."""
    return {row["name"]: hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
            for row in result.get("artifacts") or [] if row.get("path") and Path(row["path"]).is_file()}


def _decided(tmp_path: Path, *, surface: str = "", child_report: bool = True, disposition: str = "integrated",
             child_ledger: bool = False):
    data, workspace = (tmp_path / "data").resolve(), (tmp_path / "workspace").resolve()
    workspace.mkdir()
    subprocess.run(["git", "init", "--quiet", str(workspace)], check=True, capture_output=True)
    (workspace / "answer.txt").write_text("source output\n", encoding="utf-8")
    drive = headless.prepare_task_drive(data, CHILD, "empty")
    (drive / "memory" / "scratchpad.md").write_text("known one\n", encoding="utf-8")
    lineage = {"parent_task_id": PARENT, "root_task_id": PARENT, "delegation_role": "subagent"}
    task = {"id": CHILD, "drive_root": str(drive), "workspace_root": str(workspace), "memory_mode": "empty",
            **lineage, **({"task_constraint": {"surface": surface}} if surface else {})}
    write_task_result(data, CHILD, "running", child_drive_root=str(drive), workspace_root=str(workspace), **lineage)
    own = {}
    if child_report:  # the child's own registered output: copy-back places it in the canonical store
        report = artifacts.task_artifact_dir_path(drive, CHILD, create=True) / "report.md"
        report.write_text("child report\n", encoding="utf-8")
        own = {"artifacts": [artifacts.artifact_record(report, kind="report")]}
    if child_ledger:  # the child's own ledger file; host finalization rewrites the canonical copy
        ledger = artifacts.task_artifact_dir_path(drive, CHILD, create=True) / "verification_ledger.json"
        ledger.write_text(json.dumps({"schema_version": 1, "entries": [], "summary": {"total": 0}}), encoding="utf-8")
        own = {"artifacts": [*own.get("artifacts", []), artifacts.artifact_record(ledger, kind="verification_ledger")]}
    write_task_result(drive, CHILD, "completed", result="useful final text", trace_summary="trace", **lineage, **own)
    ctx = _ctx(data)
    headless.copy_child_task_result(data, task)
    _, _, early_text = _shown(ctx, data)  # the answer is readable before the host files exist
    assert "useful final text" in early_text and load_task_result(data, CHILD)["artifact_status"] == "finalizing"
    assert not headless.prepare_terminal_task_files(data, task)["error"]
    before, shown, _ = _shown(ctx, data)
    payload = {"type": "child_result_disposition", "child_task_id": CHILD,
               "disposition": disposition, "child_result_sha256": shown}
    assert _tree_note(ctx, "decision", "The parent consumed this exact result.", payload=payload).startswith("OK:")
    assert _current_child_result_disposition(load_effective_task_result(data, CHILD)) == disposition
    return data, drive, ctx, before, shown


def _retained(data: Path, drive: Path, ctx: SimpleNamespace):
    """Ordinary deferred history retention, then the child-drive custody pass."""
    headless.retry_child_task_refs(data, drive, CHILD)
    settled = task_custody.settle_child_drive(data, CHILD, drive, live=lambda _tid: False)
    after, after_hash, text = _shown(ctx, data)
    return after, after_hash, text, settled


def _host_rows(result: dict) -> list:
    return [row for row in result.get("artifacts") or [] if row.get("kind") in HOST_KINDS]


@pytest.mark.parametrize("surface", ["", "genesis"])
def test_retaining_identical_bytes_keeps_the_exact_parent_decision(tmp_path, surface):
    data, drive, ctx, before, shown = _decided(tmp_path, surface=surface)
    on_disk = _bytes_on_disk(before)
    after, after_hash, _, settled = _retained(data, drive, ctx)
    assert settled["status"] == "removed" and not drive.exists()
    assert _bytes_on_disk(after) == on_disk  # nothing changed on disk ...
    assert (after_hash, _current_child_result_disposition(after)) == (shown, "integrated")  # ... nor the decision
    rows = _host_rows(before)
    assert {row["kind"] for row in rows} == ({"workspace_patch_manifest", "memory_export"}
                                             | ({"deliverable_manifest"} if surface else set()))
    for row in rows:  # identity is published WITH the ready row, by the existing measurement owner
        raw = Path(row["path"]).read_bytes()
        assert (row.get("sha256"), row["size"]) == (hashlib.sha256(raw).hexdigest(), len(raw)), row


def test_failed_capture_manifest_carries_its_identity_and_keeps_the_decision(tmp_path, monkeypatch):
    def broken(*_args, **_kwargs):
        raise OSError("fixture capture failure")

    monkeypatch.setattr(headless, "write_workspace_patch_artifacts", broken)
    data, drive, ctx, before, shown = _decided(tmp_path, disposition="deferred")
    assert before["artifact_status"] == "failed" and "fixture capture failure" in before["artifact_error"]
    after, after_hash, _, _ = _retained(data, drive, ctx)
    assert (after_hash, _current_child_result_disposition(after)) == (shown, "deferred")
    assert after["artifact_status"] == "failed"  # the failure itself stays material and visible
    manifest = next(row for row in before["artifacts"] if row["kind"] == "workspace_patch_manifest")
    assert manifest.get("sha256") == hashlib.sha256(Path(manifest["path"]).read_bytes()).hexdigest()


@pytest.mark.parametrize("entry", ["prepare_terminal_task_files", "finalize_task_artifacts"])
def test_an_unmeasurable_failure_manifest_keeps_the_capture_error_first(tmp_path, monkeypatch, entry):
    """The capture-failure handler stays total: a read-back fault on its own manifest is
    disclosed beside the capture error, never instead of it; the supervisor's direct call,
    which only logs an escape, still settles the row. No size or digest is invented."""
    def broken(*_args, **_kwargs):
        raise OSError("fixture capture failure")

    real = store.open_for_read

    def denied(path, *args, **kwargs):
        if Path(path).name == "workspace_patch.json":
            raise PermissionError(13, "fixture read-back denial", str(path))
        return real(path, *args, **kwargs)

    data, workspace = (tmp_path / "data").resolve(), (tmp_path / "workspace").resolve()
    workspace.mkdir()
    subprocess.run(["git", "init", "--quiet", str(workspace)], check=True, capture_output=True)
    drive = headless.prepare_task_drive(data, CHILD, "empty")
    lineage = {"parent_task_id": PARENT, "root_task_id": PARENT, "delegation_role": "subagent"}
    task = {"id": CHILD, "drive_root": str(drive), "workspace_root": str(workspace), "memory_mode": "empty", **lineage}
    write_task_result(data, CHILD, "running", child_drive_root=str(drive), workspace_root=str(workspace), **lineage)
    write_task_result(drive, CHILD, "completed", result="useful final text", **lineage)
    headless.copy_child_task_result(data, task)
    monkeypatch.setattr(headless, "write_workspace_patch_artifacts", broken)
    monkeypatch.setattr(store, "open_for_read", denied)
    if entry == "prepare_terminal_task_files":
        assert not headless.prepare_terminal_task_files(data, task)["error"]
    else:  # supervisor/task_lifecycle and task_reaper call it directly and only log an escape
        headless.finalize_task_artifacts(data, task)
    result = load_task_result(data, CHILD)
    assert result["artifact_status"] == "failed"
    assert result["artifact_error"].startswith(
        "OSError: fixture capture failure; failure manifest unmeasured: PermissionError"), result["artifact_error"]
    row = next(row for row in result["artifacts"] if row["kind"] == "workspace_patch_manifest")
    assert row["measured"] is False and "size" not in row and "sha256" not in row, row
    assert "fixture read-back denial" in row["errors"][0]
    _shown(_ctx(data), data)  # the parent's real read takes the unmeasured row as it is


def test_partial_deliverable_manifest_keeps_its_warning_beside_its_identity(tmp_path, monkeypatch):
    real = store.open_for_read

    def denied(path, *args, **kwargs):
        if Path(path).name == "locked.bin":
            raise PermissionError(13, "fixture read denial", str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(store, "open_for_read", denied)
    original = headless._build_deliverable_manifest

    def with_unreadable(workspace_root, *args, **kwargs):
        (Path(workspace_root) / "locked.bin").write_bytes(b"unreadable")
        try:
            return original(workspace_root, *args, **kwargs)
        finally:
            (Path(workspace_root) / "locked.bin").unlink()

    monkeypatch.setattr(headless, "_build_deliverable_manifest", with_unreadable)
    data, drive, ctx, before, shown = _decided(tmp_path, surface="genesis")
    row = next(row for row in before["artifacts"] if row["kind"] == "deliverable_manifest")
    warning = ["Automatic workspace listing is partial: 1 read gaps."]
    assert (row["errors"], row["complete"], row["gap_count"]) == (warning, False, 1)
    after, after_hash, _, _ = _retained(data, drive, ctx)
    kept = next(row for row in after["artifacts"] if row["kind"] == "deliverable_manifest")
    assert (kept["errors"], kept["complete"], kept["gap_count"]) == (warning, False, 1)
    assert (after_hash, _current_child_result_disposition(after)) == (shown, "integrated")
    assert row.get("sha256") == hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
    assert "status" not in row  # the producer's facts are kept, not overlaid with a clean "ready" record


@pytest.mark.parametrize("change", ["equal_length_bytes", "new_child_artifact", "critical_notice",
                                    "read_error", "unregistered_listing"])
def test_real_changes_after_the_decision_still_reopen_it(tmp_path, monkeypatch, change):
    data, drive, ctx, before, shown = _decided(tmp_path)
    export = Path(next(row for row in before["artifacts"] if row["kind"] == "memory_export")["path"])
    old = export.read_bytes()
    if change == "equal_length_bytes":
        new = old.replace(b"known one", b"known two")
        assert len(new) == len(old) and new != old
        export.write_bytes(new)
    elif change == "new_child_artifact":  # staged by custody: the child store holds it, the canonical one not yet
        late = artifacts.task_artifact_dir_path(drive, CHILD, create=True) / "late-result.txt"
        late.write_text("late result from the retained child drive\n", encoding="utf-8")
        current = load_task_result(drive, CHILD)
        write_task_result(drive, CHILD, current["status"],
                          artifacts=[*(current.get("artifacts") or []), artifacts.artifact_record(late, kind="report")])
    elif change == "critical_notice":
        write_task_result(data, CHILD, load_task_result(data, CHILD)["status"],
                          terminal_host_notice="Critical limit: the required integration check did not run.")
    elif change == "read_error":
        real = store.open_for_read
        monkeypatch.setattr(store, "open_for_read", lambda path, *a, **kw: (_ for _ in ()).throw(
            PermissionError(13, "fixture read denial", str(path))) if Path(path) == export else real(path, *a, **kw))
    elif change == "unregistered_listing":  # a legitimate post-terminal file: listed, never hashed by a read
        (export.parent / "notes.txt").write_text("written after the decision\n", encoding="utf-8")
    after, after_hash, text, settled = _retained(data, drive, ctx)
    assert after_hash != shown and _current_child_result_disposition(after) == ""
    if change == "equal_length_bytes":
        row = next(row for row in after["artifacts"] if row["kind"] == "memory_export")
        assert row["sha256"] == hashlib.sha256(export.read_bytes()).hexdigest() != hashlib.sha256(old).hexdigest()
    elif change == "new_child_artifact":
        assert settled["published"] == 1
        assert any(row["name"] == "late-result.txt" and Path(row["path"]).is_relative_to(data / "task_results")
                   for row in after["artifacts"])
    elif change == "critical_notice":
        assert "Critical limit" in text
    elif change == "read_error":
        row = next(row for row in after["artifacts"] if row["kind"] == "memory_export")
        assert row["copy_status"] == "failed" and "fixture read denial" in row["copy_error"]
        assert export.read_bytes() == old
    else:
        listed = next(row for row in after["artifacts"] if row["name"] == "notes.txt")
        assert listed["measured"] is False and "sha256" not in listed


def test_a_host_rewritten_verification_ledger_keeps_its_real_byte_conflict_visible(tmp_path):
    """Measured publication hides no real difference: the child's ledger bytes and the host-refreshed
    canonical ledger differ, the parent sees the conflict, and custody publishes both versions."""
    data, drive, ctx, before, shown = _decided(tmp_path, child_ledger=True)
    conflicted = next(row for row in before["artifacts"] if row["name"] == "verification_ledger.json")
    assert conflicted["status"] == "failed" and "recorded identities conflict" in conflicted["errors"][0]
    assert "recorded identities conflict" in _get_task_result(ctx, CHILD)
    after, after_hash, _, settled = _retained(data, drive, ctx)
    ledgers = sorted(row["name"] for row in after["artifacts"] if row["kind"] == "verification_ledger")
    assert settled["published"] == 1 and len(ledgers) == 2 and "verification_ledger.json" in ledgers
    assert after_hash != shown and _current_child_result_disposition(after) == ""
