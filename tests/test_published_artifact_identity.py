"""A ready host artifact is published with its measured byte identity (#1969).

The real chain: the child finishes → copy-back exposes the early answer → terminal
file preparation publishes the host artifacts → the parent reads the exact result
hash and records a disposition on it → ordinary deferred history retention and the
child-drive custody pass. Retaining the SAME bytes keeps that exact decision; changed
bytes, a new artifact, a late host notice or a read error still reopen it. A child's
verification ledger file is such a capture too: host finalization never rewrites it (#2472).
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import artifacts, headless, outcomes, store, task_custody
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.task_status import load_effective_task_result
from ouroboros.tools.control_task_results import _get_task_result
from ouroboros.tools.join_ledger import _child_result_sha256, _current_child_result_disposition
from ouroboros.tools.recent_tasks import _handle_recent_tasks
from ouroboros.tools.task_tree import _tree_note

PARENT, CHILD = "parent1", "child1"
HOST_KINDS = {"workspace_patch_manifest", "memory_export", "deliverable_manifest"}
LINEAGE = {"parent_task_id": PARENT, "root_task_id": PARENT, "delegation_role": "subagent"}


def _task(tmp_path: Path, drive: Path | None) -> dict:
    """The child's task: its own drive (split stores) or none (it ran in the canonical store)."""
    own_drive = {"drive_root": str(drive), "memory_mode": "empty"} if drive is not None else {}
    return {"id": CHILD, "workspace_root": str((tmp_path / "workspace").resolve()), **own_drive, **LINEAGE}


def _ledger(size: str) -> dict:
    """A producer's ledger: the early failed check stays beside the later passing ones."""
    detail = "x" * (1600 if size == "omitted" else 8)
    entries = [{"kind": "verification_receipt", "status": status, "criterion_id": f"check-{index}", "detail": detail}
               for index, status in enumerate(["fail"] + ["pass"] * 8)]
    return {"schema_version": 2, "created_at": "2026-10-29T11:00:00+00:00", "task_id": CHILD,
            "entries": entries, "summary": {"entry_count": len(entries), "has_failures": True}}


def _produced(root: Path, ledger: dict) -> tuple[dict, dict]:
    """The real producer: a large ledger becomes the store's file plus an omitted stub."""
    refs = outcomes.maybe_write_verification_artifact(root, CHILD, ledger)
    return refs, {"verification_ledger": refs["inline"], "artifacts": [refs["artifact"]] if refs["artifact"] else []}


def _outcome(text: str) -> dict:
    return json.loads(text.split("[SUBTASK_OUTCOME]\n", 1)[1].split("\n[/SUBTASK_OUTCOME]", 1)[0])


def _broken_capture(*_args, **_kwargs):
    raise OSError("fixture capture failure")


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


def _workspace(tmp_path: Path) -> Path:
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    subprocess.run(["git", "init", "--quiet", str(workspace)], check=True, capture_output=True)
    (workspace / "answer.txt").write_text("source output\n", encoding="utf-8")
    return workspace


def _decided(tmp_path: Path, *, surface: str = "", child_report: bool = True, disposition: str = "integrated",
             child_ledger: bool = False, ledger: dict | None = None):
    data, workspace = (tmp_path / "data").resolve(), _workspace(tmp_path)
    drive = headless.prepare_task_drive(data, CHILD, "empty")
    (drive / "memory" / "scratchpad.md").write_text("known one\n", encoding="utf-8")
    task = {**_task(tmp_path, drive), **({"task_constraint": {"surface": surface}} if surface else {})}
    write_task_result(data, CHILD, "running", child_drive_root=str(drive), workspace_root=str(workspace), **LINEAGE)
    own = {}
    if child_report:  # the child's own registered output: copy-back places it in the canonical store
        report = artifacts.task_artifact_dir_path(drive, CHILD, create=True) / "report.md"
        report.write_text("child report\n", encoding="utf-8")
        own = {"artifacts": [artifacts.artifact_record(report, kind="report")]}
    if child_ledger:  # the child's own ledger file, published by copy-back like any child output
        path = artifacts.task_artifact_dir_path(drive, CHILD, create=True) / "verification_ledger.json"
        path.write_text(json.dumps({"schema_version": 1, "entries": [], "summary": {"total": 0}}), encoding="utf-8")
        own = {"artifacts": [*own.get("artifacts", []), artifacts.artifact_record(path, kind="verification_ledger")]}
    if ledger is not None:
        _refs, produced = _produced(drive, ledger)
        own = {**produced, "artifacts": [*own.get("artifacts", []), *produced["artifacts"]]}
    write_task_result(drive, CHILD, "completed", result="useful final text", trace_summary="trace", **LINEAGE, **own)
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


@pytest.mark.parametrize("change", ["equal_length_bytes", "read_error", "unregistered_listing"])
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


def test_host_finalization_keeps_a_published_child_ledger_and_the_parent_decision(tmp_path):
    """#2472: copy-back publishes the child's ledger with its measured identity and host finalization
    no longer rewrites those bytes, so the parent reads one identity, decides on it, and retirement
    carries no second version that would reopen that decision."""
    data, drive, ctx, before, shown = _decided(tmp_path, child_ledger=True)
    captured = (artifacts.task_artifact_dir_path(drive, CHILD) / "verification_ledger.json").read_bytes()
    row = next(row for row in before["artifacts"] if row["name"] == "verification_ledger.json")
    assert Path(row["path"]).read_bytes() == captured and row["sha256"] == hashlib.sha256(captured).hexdigest()
    assert (row["status"], row["errors"]) == ("ready", []) and before["artifact_bundle"]["status"] != "failed"
    assert "recorded identities conflict" not in _get_task_result(ctx, CHILD)
    headless.copy_child_task_result(data, _task(tmp_path, drive))  # a later publisher (the cancel lane's)
    assert "recorded identities conflict" not in _get_task_result(ctx, CHILD)
    after, after_hash, _, settled = _retained(data, drive, ctx)
    assert (settled["status"], settled["published"]) == ("removed", 0)
    assert [(kept["name"], kept["sha256"]) for kept in after["artifacts"] if kept["kind"] == "verification_ledger"] \
        == [("verification_ledger.json", row["sha256"])]
    assert (after_hash, _current_child_result_disposition(after)) == (shown, "integrated")


@pytest.mark.parametrize("writer", ["child_late_write", "canonical_rewrite"])
def test_a_real_second_ledger_identity_stays_a_visible_conflict(tmp_path, writer):
    """The detector stays whole: ledger bytes that really change after the decision, recorded by
    their writer on one store (the canonical case is what the retired finalizer did), conflict with
    the identity the other store keeps: a failed row, a reopened decision, both versions retained."""
    data, drive, ctx, before, shown = _decided(tmp_path, child_ledger=True)
    root, other = (drive, data) if writer == "child_late_write" else (data, drive)
    path = artifacts.task_artifact_dir_path(root, CHILD) / "verification_ledger.json"
    original = hashlib.sha256((artifacts.task_artifact_dir_path(other, CHILD) / path.name).read_bytes()).hexdigest()
    path.write_text(json.dumps({"schema_version": 1, "entries": [{"kind": "late", "status": "fail"}]}), encoding="utf-8")
    rewritten = hashlib.sha256(path.read_bytes()).hexdigest()
    current = load_task_result(root, CHILD)
    write_task_result(root, CHILD, current["status"], artifacts=[
        artifacts.artifact_record(path, kind="verification_ledger") if row.get("kind") == "verification_ledger" else row
        for row in current["artifacts"]])
    effective = load_effective_task_result(data, CHILD)
    row = next(row for row in effective["artifacts"] if row["name"] == "verification_ledger.json")
    assert row["status"] == "failed" and {original, rewritten} <= set(re.findall(r"[0-9a-f]{64}", row["errors"][-1]))
    assert effective["artifact_bundle"]["status"] == "failed"
    assert "recorded identities conflict" in _get_task_result(ctx, CHILD)
    after, after_hash, _, settled = _retained(data, drive, ctx)
    assert settled["published"] == 1
    assert sorted(kept["sha256"] for kept in after["artifacts"] if kept["kind"] == "verification_ledger") \
        == sorted([original, rewritten])
    assert after_hash != shown and _current_child_result_disposition(after) == ""


def _readers(ctx: SimpleNamespace) -> list:
    """The three projections: the full handoff, the evidence view and the recent-task listing."""
    outcome = _outcome(_get_task_result(ctx, CHILD))
    listed = next(row for row in json.loads(_handle_recent_tasks(ctx, limit=5))["tasks"] if row["task_id"] == CHILD)
    return [outcome["verification_ledger"], json.loads(_get_task_result(ctx, CHILD, view="evidence"))["verification_ledger"],
            listed["verification_ledger"]]


def _capture_facts(ledger: dict, omitted: bool) -> dict:
    """The capture's own observation, dated by its creation."""
    return {"created_at": ledger["created_at"], "summary": ledger["summary"],
            "entry_count": ledger["summary"]["entry_count"]} | ({"omitted_to_artifact": True} if omitted else {})


def _refreshed_facts(data: Path, ledger: dict, count: int, source: str) -> dict:
    """The capture's date beside the summary the last finalization refreshed with the current
    artifact status, and for an omitted stub the complete file it was computed from."""
    facts = {"created_at": ledger["created_at"], "entry_count": count,
             "summary": {"entry_count": count, "has_failures": True},
             "summary_refreshed_at": load_task_result(data, CHILD)["artifact_finalized_at"]}
    return facts | ({"omitted_to_artifact": True, "summary_source_sha256": source} if source else {})


_REFRESH_FACTS = {"summary_refreshed_at", "summary_source_sha256"}


def _shown_by_readers(ctx: SimpleNamespace, facts: dict, absent: set = frozenset()) -> bool:
    return all({key: shown.get(key) for key in facts} == facts and not absent & shown.keys()
               for shown in _readers(ctx))


@pytest.mark.parametrize(("capture", "size"), [("clean", "omitted"), ("failed", "inline")])
def test_a_child_ledger_summary_is_current_while_its_capture_stays_exact(tmp_path, monkeypatch, size, capture):
    """Split stores: the canonical row's summary folds in the current artifact status (an omitted
    stub reads it from the complete file, never from itself) while the published capture, earlier
    failed check included, keeps its bytes. Every reader shows that refreshed summary with its date
    and source: the child's copy of the same capture does not replace it while its drive is served."""
    if capture == "failed":
        monkeypatch.setattr(headless, "write_workspace_patch_artifacts", _broken_capture)
    ledger = _ledger(size)
    data, drive, ctx, before, shown = _decided(tmp_path, ledger=ledger)
    stored, count = load_task_result(data, CHILD), len(ledger["entries"]) + (capture == "failed")
    assert stored["artifact_bundle"]["status"] == ("failed" if capture == "failed" else "ready_with_changes")
    files = [row for row in before["artifacts"] if row["kind"] == "verification_ledger"]
    identities = [(row["sha256"], row["status"]) for row in files]
    captured = (artifacts.task_artifact_dir_path(drive, CHILD) / "verification_ledger.json").read_bytes() \
        if size == "omitted" else b""
    if size == "omitted":
        assert identities == [(hashlib.sha256(captured).hexdigest(), "ready")]
        assert json.loads(captured)["summary"] == ledger["summary"] and "entries" not in stored["verification_ledger"]
    else:
        assert not files and stored["verification_ledger"]["entries"][0]["status"] == "fail"
    source = files[0]["sha256"] if files else ""
    refreshed = _refreshed_facts(data, ledger, count, source)
    assert {key: stored["verification_ledger"].get(key) for key in refreshed if key != "entry_count"} \
        == {key: value for key, value in refreshed.items() if key != "entry_count"}
    assert _shown_by_readers(ctx, refreshed)
    assert _outcome(_get_task_result(ctx, CHILD))["artifact_bundle"]["status"] == stored["artifact_bundle"]["status"]

    def unchanged() -> bool:  # one identity per published ledger, holding the captured bytes
        rows = load_effective_task_result(data, CHILD)["artifacts"]
        return ([(row["sha256"], row["status"]) for row in rows if row["kind"] == "verification_ledger"] == identities
                and all(Path(row["path"]).read_bytes() == captured for row in files))

    after, after_hash, _, settled = _retained(data, drive, ctx)
    assert (settled["status"], settled["published"], after_hash) == ("removed", 0, shown) and unchanged()
    assert _shown_by_readers(ctx, refreshed)
    headless.finalize_task_artifacts(data, _task(tmp_path, drive))  # a repeated pass after retirement
    assert unchanged() and _shown_by_readers(ctx, _refreshed_facts(data, ledger, count, source))


def _built(size: str, artifact_status: str, tool_calls: tuple = ()) -> dict:
    """The runtime's own ledger builder. A long task contract, a supported input, omits the capture
    without any failed check; the artifact status it sees at build time becomes its own entry."""
    contract = {"status": "recorded", "objective": "task", "expected_output": "x" * (13000 if size == "omitted" else 8)}
    return outcomes.build_verification_ledger(
        task={"id": CHILD, "task_contract": contract},
        loop_outcome={"outcome_axes": {"execution": {"status": "ok"}, "objective": {"status": "pass"}}},
        llm_trace={"tool_calls": list(tool_calls)}, artifact_bundle={"status": artifact_status})


def _row_facts(ledger: dict) -> dict:
    return {key: ledger[key] for key in ("created_at", "summary", *_REFRESH_FACTS) if key in ledger}


@pytest.mark.parametrize(("transition", "size"), [
    ("clean_then_capture_failed", "omitted"),
    ("pending_then_ready", "inline"),
])
def test_readers_and_a_repeated_copyback_keep_the_refreshed_summary(tmp_path, monkeypatch, size, transition):
    """The child's capture contradicts the current artifact status: clean before a failed host capture,
    or pending before a ready one. Its copy in the child's row is the same capture, so neither the served
    drive (full result, evidence, recent tasks) nor a repeated copy-back (the cancel lane publishes
    again) brings it back over the refreshed summary, and retirement keeps that summary."""
    failed = transition == "clean_then_capture_failed"
    if failed:
        monkeypatch.setattr(headless, "write_workspace_patch_artifacts", _broken_capture)
    ledger = _built(size, "ready" if failed else "pending")
    assert ledger["summary"]["has_failures"] is not failed
    data, drive, ctx, _before, shown = _decided(tmp_path, ledger=ledger)
    refreshed = load_task_result(data, CHILD)["verification_ledger"]
    assert refreshed["summary"]["has_failures"] is failed and refreshed["summary_refreshed_at"]
    assert ("summary_source_sha256" in refreshed) == (size == "omitted")
    assert _shown_by_readers(ctx, _row_facts(refreshed))
    headless.copy_child_task_result(data, _task(tmp_path, drive))
    assert load_task_result(data, CHILD)["verification_ledger"] == refreshed
    after, after_hash, _, settled = _retained(data, drive, ctx)
    assert (settled["status"], after_hash, after["verification_ledger"]) == ("removed", shown, refreshed)
    assert _shown_by_readers(ctx, _row_facts(refreshed))


@pytest.mark.parametrize("size", ["omitted", "inline"])
def test_a_later_child_capture_replaces_the_refreshed_summary(tmp_path, size):
    """The canonical row keeps only its own capture: a new producer call in the child's drive, dated by
    itself and holding a new failure, reaches every reader at once and the canonical row with the next
    copy-back, undated by the refresh of the earlier capture; a rewritten ledger file meanwhile reads
    as an identity conflict."""
    data, drive, ctx, _before, _shown = _decided(tmp_path, ledger=_built(size, "ready"))
    assert load_task_result(data, CHILD)["verification_ledger"]["summary"]["has_failures"] is False
    later = _built(size, "ready", tool_calls=({"tool": "run_shell", "status": "error", "is_error": True},))
    _refs, produced = _produced(drive, later)
    current = load_task_result(drive, CHILD)
    write_task_result(drive, CHILD, current["status"], verification_ledger=produced["verification_ledger"], artifacts=[
        *(row for row in current["artifacts"] if row["kind"] != "verification_ledger"), *produced["artifacts"]])
    facts = _capture_facts(later, size == "omitted")
    assert later["summary"]["has_failures"] is True and _shown_by_readers(ctx, facts, absent=_REFRESH_FACTS)
    assert ("recorded identities conflict" in _get_task_result(ctx, CHILD)) == (size == "omitted")
    headless.copy_child_task_result(data, _task(tmp_path, drive))
    copied = load_task_result(data, CHILD)["verification_ledger"]
    assert _row_facts(copied) == _row_facts(produced["verification_ledger"])
    assert _retained(data, drive, ctx)[3]["status"] == "removed"
    assert _shown_by_readers(ctx, facts, absent=_REFRESH_FACTS)


@pytest.mark.parametrize(("capture", "size"), [("clean", "omitted"), ("failed", "inline")])
def test_a_same_store_ledger_keeps_its_capture_across_finalizations(tmp_path, monkeypatch, size, capture):
    """One store (no child drive), without copy-back: each finalization recomputes the summary from
    the unchanged capture, so a second pass neither rewrites the file nor counts its status twice."""
    if capture == "failed":
        monkeypatch.setattr(headless, "write_workspace_patch_artifacts", _broken_capture)
    data, ledger, task = (tmp_path / "data").resolve(), _ledger(size), _task(tmp_path, None)
    _workspace(tmp_path)
    refs, produced = _produced(data, ledger)
    write_task_result(data, CHILD, "completed", result="useful final text", workspace_root=task["workspace_root"],
                      **LINEAGE, **produced)
    assert (refs["artifact"] is not None) == (size == "omitted")
    captured = Path(refs["artifact"]["path"]).read_bytes() if refs["artifact"] else b""
    source, count = (refs["artifact"] or {}).get("sha256", ""), len(ledger["entries"]) + (capture == "failed")
    assert not headless.prepare_terminal_task_files(data, task)["error"]  # the real terminal preparation
    for repeated in (False, True):
        if repeated:
            headless.finalize_task_artifacts(data, task)
        rows = [row for row in load_effective_task_result(data, CHILD)["artifacts"] if row["kind"] == "verification_ledger"]
        assert [(row["sha256"], row["status"]) for row in rows] == ([(source, "ready")] if source else [])
        assert all(Path(row["path"]).read_bytes() == captured for row in rows)
        assert _shown_by_readers(_ctx(data), _refreshed_facts(data, ledger, count, source))


@pytest.mark.parametrize("damage", ["missing", "changed_bytes"])
def test_an_unverifiable_ledger_file_leaves_the_dated_capture_summary(tmp_path, damage):
    """No summary from bytes that are not the recorded capture and none rebuilt from the stub: the
    capture's own summary stays, undated by any refresh, and the row is never re-measured over it."""
    data, ledger = (tmp_path / "data").resolve(), _ledger("omitted")
    _workspace(tmp_path)
    refs, produced = _produced(data, ledger)
    path = Path(refs["artifact"]["path"])
    if damage == "missing":
        path.unlink()
    else:  # equal length, different bytes: recomputed, it would read clean
        path.write_bytes(path.read_bytes().replace(b'"fail"', b'"pass"'))
    damaged = path.read_bytes() if path.exists() else None
    write_task_result(data, CHILD, "completed", result="useful final text", workspace_root=str(tmp_path / "workspace"),
                      **LINEAGE, **produced)
    assert not headless.prepare_terminal_task_files(data, _task(tmp_path, None))["error"]
    stored = load_task_result(data, CHILD)
    assert stored["verification_ledger"] == refs["inline"]
    row = next(row for row in stored["artifacts"] if row["kind"] == "verification_ledger")
    assert (row["sha256"], row["size"]) == (refs["artifact"]["sha256"], refs["artifact"]["size"])
    assert (path.read_bytes() if path.exists() else None) == damaged
    shown = _outcome(_get_task_result(_ctx(data), CHILD))["verification_ledger"]
    assert shown["summary"] == ledger["summary"] and "summary_refreshed_at" not in shown
    assert "summary_source_sha256" not in shown and shown["created_at"] == ledger["created_at"]
