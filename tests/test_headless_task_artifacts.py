"""Child result copyback, artifact endpoints and task-drive lifecycle.

Split verbatim out of ``tests/test_headless_cli.py`` by theme. This module
owns what happens to a finished task's artifacts: copyback accounting and
acceptance markers, the artifact-serving endpoint, memory export, startup
pruning of terminal drives/scratch, and external child budget state.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from types import SimpleNamespace

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.gateway import task_archive
from ouroboros.gateway.tasks import (
    api_task_artifact,
)
from ouroboros.headless import (
    ARTIFACT_STATUS_FINALIZING,
    ARTIFACT_STATUS_READY,
    ARTIFACT_STATUS_READY_WITH_CHANGES,
    build_memory_export,
    finalize_task_artifacts,
    prune_headless_task_drives,
    prune_task_drives,
    task_artifacts_dir,
)
from ouroboros.task_results import write_task_result


from tests._headless_cli_shared import (  # noqa: F401  (autouse fixture applies on import)
    _init_repo_with_file,
    _managed_worker_pool_available,
)


def _assert_file_response(response, content: bytes) -> None:
    # Owner-approved #1297: platforms without confined opens return a typed 503.
    if task_archive.CONFINED:
        assert response.status_code == 200
        assert response.content == content
    else:
        assert response.status_code == 503
        assert response.json()["reason_code"] == "artifact_unavailable"


def test_copy_child_result_cannot_overwrite_finalized_accounting(tmp_path):
    """F2: once the root's terminal checkpoint has finalized accounting
    (task_cost_finalized rides the same write as post_task_synthesis), a late
    headless-mirror copy-back may still enrich the result but the parent-owned
    cost/round/token fields stay finalized (the saga displayed the $66 root-only
    mirror cost instead of the $128 finalized subtree total)."""
    from ouroboros.headless import copy_child_task_result
    from ouroboros.task_results import STATUS_COMPLETED

    parent = tmp_path / "data"
    child = tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    task_id = "costfinal"
    write_task_result(
        parent, task_id, STATUS_COMPLETED,
        result="root done",
        root_phase_checkpoint={"post_task_synthesis": "completed"},
        cost_usd=127.97, cost_final=True,
        cost_usd_with_children=127.97, cost_with_children_partial=False,
        total_rounds=200, prompt_tokens=1000, completion_tokens=500,
    )
    write_task_result(
        child, task_id, STATUS_COMPLETED,
        result="mirror done",
        cost_usd=66.30, cost_final=True,
        cost_usd_with_children=66.30, cost_with_children_partial=True,
        total_rounds=150, prompt_tokens=700, completion_tokens=300,
        mirror_only_fact="from-child",
    )

    merged = copy_child_task_result(parent, {"id": task_id, "drive_root": str(child)})

    assert merged is not None
    # ABI-3 fix-round-2: finalized accounting stays parent-owned AND is
    # persisted/merged under the honest names only.
    assert merged["accounted_upper_bound_usd"] == 127.97
    assert merged["accounted_upper_bound_usd_with_children"] == 127.97
    assert "cost_usd" not in merged and "cost_usd_with_children" not in merged
    assert merged["cost_with_children_partial"] is False
    assert merged["total_rounds"] == 200
    assert merged["prompt_tokens"] == 1000
    assert merged["completion_tokens"] == 500
    # Non-accounting enrichment from the child mirror still lands.
    assert merged["mirror_only_fact"] == "from-child"
    assert merged["result"] == "mirror done"
    assert merged["root_phase_checkpoint"]["post_task_synthesis"] == "completed"


def test_copy_child_result_merges_cost_before_finalization(tmp_path):
    """Before the terminal checkpoint finalizes accounting, the child mirror's
    cost projection is still the freshest fact and must keep flowing."""
    from ouroboros.headless import copy_child_task_result
    from ouroboros.task_results import STATUS_COMPLETED

    parent = tmp_path / "data"
    child = tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    task_id = "costlive"
    write_task_result(parent, task_id, STATUS_COMPLETED, result="root running")
    write_task_result(
        child, task_id, STATUS_COMPLETED,
        result="mirror done", cost_usd=12.5, total_rounds=42,
    )

    merged = copy_child_task_result(parent, {"id": task_id, "drive_root": str(child)})

    assert merged is not None
    # ABI-3 fix-round-2: the write seam persists the honest name only, even
    # for a legacy-spelled writer (deprecated-wins, then stripped).
    assert merged["accounted_upper_bound_usd"] == 12.5
    assert "cost_usd" not in merged
    assert merged["total_rounds"] == 42


def test_effective_result_preserves_workspace_artifact_status_with_child_drive(tmp_path):
    from ouroboros.headless import copy_child_task_result
    from ouroboros.task_results import STATUS_COMPLETED
    from ouroboros.task_status import load_effective_task_result

    parent = tmp_path / "data"
    child = tmp_path / "child"
    repo = tmp_path / "repo"
    parent.mkdir()
    child.mkdir()
    _init_repo_with_file(repo)
    old_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    (repo / "tracked.txt").write_text("new\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=T", "commit", "-m", "move"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    task_id = "patchfail"
    write_task_result(
        child,
        task_id,
        STATUS_COMPLETED,
        result="child done",
        artifact_status=ARTIFACT_STATUS_READY,
        artifact_bundle={"status": ARTIFACT_STATUS_READY, "artifacts": [], "errors": []},
        ts="2026-01-01T00:00:02Z",
    )
    ledger_path = parent / "task_results" / "artifacts" / task_id / "verification_ledger.json"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(
        json.dumps({
            "schema_version": 2,
            "outcome_axes": {
                "artifacts": {"status": "finalizing"},
                "objective": {"status": "not_evaluated", "source": "none"},
            },
            "entries": [{"kind": "objective_outcome", "status": "not_evaluated"}],
        }),
        encoding="utf-8",
    )
    captured_ledger = ledger_path.read_bytes()
    write_task_result(
        parent,
        task_id,
        STATUS_COMPLETED,
        result="child done",
        workspace_root=str(repo),
        child_drive_root=str(child),
        artifact_status="finalizing",
        artifacts=[{"kind": "verification_ledger", "name": "verification_ledger.json", "path": str(ledger_path)}],
        child_status=STATUS_COMPLETED,
    )

    finalize_task_artifacts(
        parent,
        {
            "id": task_id,
            "workspace_root": str(repo),
            "drive_root": str(child),
            "metadata": {"workspace_preflight": {"git": {"head": old_head}}},
        },
    )

    effective = load_effective_task_result(parent, task_id)
    assert effective["artifact_status"] == ARTIFACT_STATUS_READY_WITH_CHANGES
    assert not effective.get("artifact_error")
    assert effective["artifact_bundle"]["status"] == ARTIFACT_STATUS_READY_WITH_CHANGES
    # The ledger file stays its dated capture, "finalizing" word included (#2472): the current
    # status is the row's bundle above, never a rewrite of published bytes.
    assert ledger_path.read_bytes() == captured_ledger

    copied = copy_child_task_result(parent, {"id": task_id, "workspace_root": str(repo), "drive_root": str(child)})
    assert copied is not None
    assert copied["artifact_status"] == ARTIFACT_STATUS_READY_WITH_CHANGES
    assert not copied.get("artifact_error")
    assert copied["artifact_bundle"]["status"] == ARTIFACT_STATUS_READY_WITH_CHANGES

    readonly_task_id = "readonlychild"
    write_task_result(
        child,
        readonly_task_id,
        STATUS_COMPLETED,
        result="readonly handoff",
        workspace_root=str(repo),
        workspace_mode="external",
        delegation_role="subagent",
        task_constraint={"mode": "local_readonly_subagent"},
    )
    copied_readonly = copy_child_task_result(
        parent,
        {
            "id": readonly_task_id,
            "workspace_root": str(repo),
            "drive_root": str(child),
            "delegation_role": "subagent",
            "task_constraint": {"mode": "local_readonly_subagent"},
        },
    )
    assert copied_readonly is not None
    assert copied_readonly.get("artifact_status", "") != "finalizing"
    assert "child_status" not in copied_readonly
    effective_readonly = load_effective_task_result(parent, readonly_task_id)
    assert effective_readonly["status"] == STATUS_COMPLETED
    assert effective_readonly["workspace_root"] == str(repo)


def test_child_copyback_preserves_acceptance_verdict_and_terminal_post_task_marker(tmp_path):
    from ouroboros.headless import copy_child_task_result
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result

    parent = tmp_path / "data"
    child = tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    task_id = "root-checkpoint"
    write_task_result(
        child,
        task_id,
        STATUS_COMPLETED,
        root_phase_checkpoint={
            "phase": "task_acceptance",
            "status": "degraded",
            "pass_index": 2,
            "post_task_synthesis": "pending_once",
        },
    )
    write_task_result(
        parent,
        task_id,
        STATUS_COMPLETED,
        root_phase_checkpoint={
            "phase": "task_acceptance",
            "status": "not_required",
            "pass_index": 0,
            "post_task_synthesis": "completed",
        },
    )

    copied = copy_child_task_result(parent, {"id": task_id, "drive_root": str(child)})

    assert copied is not None
    assert copied["root_phase_checkpoint"] == {
        "phase": "task_acceptance",
        "status": "degraded",
        "pass_index": 2,
        "post_task_synthesis": "completed",
    }


def test_finalize_task_artifacts_preserves_existing_artifact_axis_fields(tmp_path):
    from ouroboros.cli import _is_terminal_result
    from ouroboros.task_results import STATUS_COMPLETED, load_task_result

    parent = tmp_path / "data"
    repo = tmp_path / "repo"
    parent.mkdir()
    _init_repo_with_file(repo)
    (repo / "tracked.txt").write_text("new\n", encoding="utf-8")
    task_id = "axisfields"
    write_task_result(
        parent,
        task_id,
        STATUS_COMPLETED,
        workspace_root=str(repo),
        artifact_status=ARTIFACT_STATUS_FINALIZING,
        artifact_bundle={"schema_version": 1, "status": "pending", "artifacts": [], "errors": []},
        outcome_axes={
            "lifecycle": {"status": STATUS_COMPLETED},
            "artifacts": {
                "status": ARTIFACT_STATUS_FINALIZING,
                "diagnostics": {"existing": True},
                "error_count": 0,
            },
            "objective": {"status": "not_evaluated", "source": "none"},
        },
    )

    finalize_task_artifacts(parent, {"id": task_id, "workspace_root": str(repo)})

    result = load_task_result(parent, task_id)
    artifact_axis = result["outcome_axes"]["artifacts"]
    assert artifact_axis["status"] == result["artifact_bundle"]["status"]
    assert result["artifact_bundle"]["status"] == result["artifact_status"]
    assert result["artifact_bundle"]["status"] not in {"pending", "finalizing"}
    assert _is_terminal_result(result) is True
    assert artifact_axis["diagnostics"] == {"existing": True}
    assert artifact_axis["error_count"] == 0


def test_a_failed_terminal_file_preparation_stamps_the_artifact_axis_with_its_bundle(tmp_path, monkeypatch):
    """A settled row whose file preparation fails is stamped ``failed`` as the finalizer stamps a capture: the bundle
    and the outcome axis take that status over the live row, and the axes keep their other fields."""
    from ouroboros import headless
    from ouroboros.task_results import STATUS_COMPLETED, load_task_result

    parent, repo = tmp_path / "data", tmp_path / "repo"
    parent.mkdir()
    _init_repo_with_file(repo)
    write_task_result(parent, "stamped", STATUS_COMPLETED, workspace_root=str(repo), outcome_axes={
        "artifacts": {"status": ARTIFACT_STATUS_FINALIZING, "diagnostics": {"existing": True}},
        "objective": {"status": "not_evaluated", "source": "none"}})

    def broken(*_args, **_kwargs):
        raise RuntimeError("capture broke")

    monkeypatch.setattr(headless, "finalize_task_artifacts", broken)
    report = headless.prepare_terminal_task_files(parent, {"id": "stamped", "workspace_root": str(repo)})
    row = load_task_result(parent, "stamped")
    assert report["terminal_source_present"] is True and "capture broke" in report["error"]
    assert (row["artifact_status"], row["artifact_bundle"]["status"]) == ("failed", "failed")
    assert row["outcome_axes"]["artifacts"] == {"status": "failed", "diagnostics": {"existing": True}}
    assert row["outcome_axes"]["objective"] == {"status": "not_evaluated", "source": "none"}


def test_effective_result_preserves_workspace_patch_kind_with_child_drive(tmp_path):
    from ouroboros.artifacts import copy_file_to_task_artifacts
    from ouroboros.cli import _patch_from_result
    from ouroboros.task_results import STATUS_COMPLETED
    from ouroboros.task_status import load_effective_task_result

    parent = tmp_path / "data"
    child = tmp_path / "child"
    repo = tmp_path / "repo"
    parent.mkdir()
    child.mkdir()
    _init_repo_with_file(repo)
    (repo / "tracked.txt").write_text("new\n", encoding="utf-8")

    task_id = "patchkind"
    report = tmp_path / "report.html"
    report.write_text("<h1>done</h1>", encoding="utf-8")
    child_record = copy_file_to_task_artifacts(SimpleNamespace(drive_root=child, task_id=task_id), report, kind="user_file")
    assert child_record is not None
    write_task_result(
        child,
        task_id,
        STATUS_COMPLETED,
        result="child done",
        artifacts=[child_record],
        artifact_status=ARTIFACT_STATUS_READY,
        ts="2026-01-01T00:00:02Z",
    )
    write_task_result(
        parent,
        task_id,
        STATUS_COMPLETED,
        result="child done",
        workspace_root=str(repo),
        child_drive_root=str(child),
        artifacts=[child_record],
        artifact_status="finalizing",
        child_status=STATUS_COMPLETED,
    )

    finalize_task_artifacts(parent, {"id": task_id, "workspace_root": str(repo), "drive_root": str(child)})

    effective = load_effective_task_result(parent, task_id)
    patch_artifacts = [
        item
        for item in effective.get("artifacts") or []
        if isinstance(item, dict) and item.get("name") == "workspace.patch"
    ]
    assert patch_artifacts
    assert patch_artifacts[0]["kind"] == "workspace_patch"
    assert any(item.get("kind") == "user_file" for item in effective.get("artifacts") or [] if isinstance(item, dict))

    class FakeClient:
        def __init__(self):
            self.paths = []

        def get_bytes(self, path):
            self.paths.append(path)
            return b"diff --git a/tracked.txt b/tracked.txt\n"

    client = FakeClient()
    assert _patch_from_result(client, task_id, effective, strict=True).startswith("diff --git")
    assert client.paths == [f"/api/tasks/{task_id}/artifacts/workspace.patch"]


def test_repeated_directory_capture_keeps_its_registered_output_store(tmp_path):
    from hashlib import sha256
    from ouroboros.artifacts import copy_file_to_task_artifacts
    from ouroboros.headless import current_workspace_manifest
    from ouroboros.task_results import load_task_result
    data, folder = tmp_path / "data", tmp_path / "folder"
    data.mkdir(); folder.mkdir()
    task = {"id": "direct-folder", "workspace_root": str(folder), "workspace_mode": "external"}
    ctx = SimpleNamespace(drive_root=data, task_id=task["id"])
    write_task_result(data, task["id"], "completed", workspace_root=str(folder), workspace_mode="external")
    for content in (b"first direct output", b"updated direct output"):
        path = folder / "output.txt"
        path.write_bytes(content)
        assert copy_file_to_task_artifacts(ctx, path, kind="user_file")
        finalize_task_artifacts(data, task)
        result = load_task_result(data, task["id"])
        source = current_workspace_manifest(data, task["id"], result)
        assert source == task_artifacts_dir(data, task["id"]) / "workspace_patch.json"
        manifest = json.loads(source.read_bytes())
        assert manifest["capture_kind"] == "directory_direct" and manifest["apply_state"] == "already_applied"
        assert any(row["sha256"] == sha256(content).hexdigest() for row in manifest["registered_outputs"])
        assert result["artifact_status"] == result["artifact_bundle"]["status"] == "ready"


def test_a_recapture_of_an_unfinished_row_is_a_version_beside_the_published_one(tmp_path):
    """An unfinished row's published capture is never rewritten in place (a withdrawn closure's, a kill's, a relaunch
    before its finish, #3248), on every surface: the next capture is a version and the first one's files keep their
    bytes at their addresses, a genesis listing among them as history. A settled row's repeat stays in place (the test
    above)."""
    import hashlib
    import pathlib

    from ouroboros.headless import current_workspace_manifest
    from ouroboros.task_results import load_task_result

    def digests(row):
        return {item["path"]: hashlib.sha256(pathlib.Path(item["path"]).read_bytes()).hexdigest()
                for item in row["artifacts"] if item.get("path")}

    for surface in ("git", "folder", "genesis"):
        data, workspace = tmp_path / surface / "data", tmp_path / surface / "workspace"
        data.mkdir(parents=True)
        fields = {"workspace_root": str(workspace), "workspace_mode": "external" if surface == "folder" else "self_worktree"}
        if surface == "folder":
            workspace.mkdir()
        else:
            _init_repo_with_file(workspace)
        if surface == "genesis":  # an acting surface needs its recorded original Git basis
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=workspace, check=True, capture_output=True, text=True)
            fields["task_constraint"] = {"mode": "acting_subagent", "surface": "genesis", "write_root": str(workspace),
                                         "base_sha": head.stdout.strip()}
        task = {"id": f"recaptured-{surface}", **fields}
        write_task_result(data, task["id"], "scheduled", **fields)
        (workspace / "out.txt").write_text("first\n", encoding="utf-8")
        finalize_task_artifacts(data, task)
        first = digests(load_task_result(data, task["id"]))
        assert first and current_workspace_manifest(data, task["id"], load_task_result(data, task["id"])).parent == \
            task_artifacts_dir(data, task["id"]), surface
        (workspace / "out.txt").write_text("second\n", encoding="utf-8")
        finalize_task_artifacts(data, task)
        row = load_task_result(data, task["id"])
        assert {path: digests(row).get(path) for path in first} == first, surface  # same bytes, same addresses
        version = current_workspace_manifest(data, task["id"], row).parent
        assert version.parent.name == "captures", surface
        assert row["artifact_status"] in {"ready", "ready_with_changes"}, (surface, row["artifact_status"])
        listings = [item["path"] for item in row["artifacts"] if item.get("kind") == "deliverable_manifest"]
        assert listings == ([str(version / "deliverable_manifest.json")] if surface == "genesis" else []), surface


def test_a_directory_recapture_hashes_no_earlier_version_again(tmp_path, monkeypatch):
    """Each directory capture holds its registered outputs in its own version (#3248), and its discovery reads the
    registrations, never the store's nested files: a later recapture of an unfinished folder task hashes no earlier
    version again, so its cost under the task's custody does not grow with the history."""
    from ouroboros import artifacts
    from ouroboros.artifacts import copy_file_to_task_artifacts
    from ouroboros.headless import current_workspace_manifest
    from ouroboros.task_results import load_task_result

    data, folder = tmp_path / "data", tmp_path / "folder"
    data.mkdir()
    folder.mkdir()
    task = {"id": "direct-versions", "workspace_root": str(folder), "workspace_mode": "external"}
    write_task_result(data, task["id"], "running", workspace_root=str(folder), workspace_mode="external")
    (folder / "output.txt").write_bytes(b"direct output")
    assert copy_file_to_task_artifacts(SimpleNamespace(drive_root=data, task_id=task["id"]), folder / "output.txt")
    real, hashed, counts = artifacts.stream_artifact_file, [], []
    monkeypatch.setattr(artifacts, "stream_artifact_file", lambda *a, **k: hashed.append(1) or real(*a, **k))
    for _ in range(3):
        hashed.clear()
        finalize_task_artifacts(data, task)
        counts.append(len(hashed))
        manifest = current_workspace_manifest(data, task["id"], load_task_result(data, task["id"]))
        assert [item["name"] for item in json.loads(manifest.read_bytes())["registered_outputs"]] == ["output.txt"]
    assert manifest.parent.parent.name == "captures" and counts[2] == counts[1], counts


def test_a_settled_rows_repeat_after_a_version_never_rewrites_a_published_address(tmp_path):
    """v1 at the base, an entered execution's reopen and its finish's v2 in ``captures/<id>``, then the row settled and ready:
    terminal preparation leaves it alone, and a direct repeat of the capture is a version too, so neither the
    historical v1 at the base nor the current v2 is rewritten or dropped (#3248). The in-place repeat stays only
    for a row whose capture was never versioned (the directory test above)."""
    import hashlib
    import pathlib

    from ouroboros.headless import current_workspace_manifest, prepare_terminal_task_files, reopen_capture_at_entry
    from ouroboros.task_results import load_task_result

    data, workspace = tmp_path / "data", tmp_path / "workspace"
    data.mkdir()
    _init_repo_with_file(workspace)
    fields = {"workspace_root": str(workspace), "workspace_mode": "self_worktree"}
    task = {"id": "repeated", **fields}

    def published(row):
        return {item["path"]: hashlib.sha256(pathlib.Path(item["path"]).read_bytes()).hexdigest()
                for item in row["artifacts"] if item.get("path")}

    write_task_result(data, task["id"], "scheduled", **fields)
    (workspace / "tracked.txt").write_text("v1\n", encoding="utf-8")
    finalize_task_artifacts(data, task)  # the closure's capture of the unfinished row: v1 at the base
    assert current_workspace_manifest(data, task["id"], load_task_result(data, task["id"])).parent == \
        task_artifacts_dir(data, task["id"])
    write_task_result(data, task["id"], "running")  # a resumed execution starts, then enters
    reopen_capture_at_entry(data, task["id"])
    assert load_task_result(data, task["id"])["artifact_status"] == "pending"
    (workspace / "tracked.txt").write_text("v2\n", encoding="utf-8")
    write_task_result(data, task["id"], "completed", result="done")
    assert prepare_terminal_task_files(data, task)["error"] == ""  # its finish: v2 in captures/<id>
    settled = load_task_result(data, task["id"])
    v2_manifest = current_workspace_manifest(data, task["id"], settled)
    assert settled["artifact_status"] == "ready_with_changes" and v2_manifest.parent.parent.name == "captures"
    before = published(settled)
    (workspace / "tracked.txt").write_text("v3\n", encoding="utf-8")
    assert prepare_terminal_task_files(data, task)["error"] == ""  # settled and ready: nothing is captured again
    assert load_task_result(data, task["id"])["artifacts"] == settled["artifacts"]
    finalize_task_artifacts(data, task)  # a direct repeat over the settled, versioned capture
    row = load_task_result(data, task["id"])
    after = published(row)
    assert {path: after.get(path) for path in before} == before  # every published address keeps its bytes
    current = current_workspace_manifest(data, task["id"], row)
    assert current.parent.parent.name == "captures" and current.parent != v2_manifest.parent
    assert "+v3" in (current.parent / "workspace.patch").read_text(encoding="utf-8")


def _entering(tmp_path, task_id):
    """An unfinished Git task whose ready capture a withdrawn closure took, started again: its row and that manifest."""
    from ouroboros.headless import current_workspace_manifest
    from ouroboros.task_results import load_task_result

    data, workspace = tmp_path / "data", tmp_path / "workspace"
    data.mkdir()
    _init_repo_with_file(workspace)
    fields = {"workspace_root": str(workspace), "workspace_mode": "self_worktree"}
    write_task_result(data, task_id, "scheduled", **fields)
    (workspace / "tracked.txt").write_text("closed\n", encoding="utf-8")
    finalize_task_artifacts(data, {"id": task_id, **fields})
    write_task_result(data, task_id, "running")
    taken = load_task_result(data, task_id)
    assert taken["artifact_status"] == ARTIFACT_STATUS_READY_WITH_CHANGES
    return data, taken, current_workspace_manifest(data, task_id, taken)


def _reopened(row, manifest):
    return row["status"] == "running" and row["artifact_status"] == "pending" and str(manifest) in {
        item["path"] for item in row["artifacts"] if item.get("kind") == "workspace_capture_history"}


def test_an_entry_waits_out_another_publisher_before_it_reopens_the_capture(tmp_path, monkeypatch):
    """#3248: another publisher holds the task's custody lock when its execution enters. The entry waits, attempt
    after attempt, neither skipping the reopen nor failing its start; the row keeps its ready capture while the lock
    is held, and once it is released that capture is history and a new one is owed."""
    import contextlib
    import threading

    from ouroboros import headless, task_custody
    from ouroboros.task_results import load_task_result

    data, taken, manifest = _entering(tmp_path, "contended")
    real_lock, refused, errors = task_custody.task_custody_lock, [], []
    holding, release, contended = threading.Event(), threading.Event(), threading.Event()

    @contextlib.contextmanager
    def observed(*args, **kwargs):  # the entry's attempts; the other publisher holds the real lock
        with real_lock(*args, **kwargs) as held:
            if not held:
                refused.append(True)
                if len(refused) >= 2:
                    contended.set()
            yield held

    def publisher():
        with real_lock(data, "contended", timeout_sec=5) as held:
            if held:
                holding.set()
                release.wait(30)

    def enter():
        try:
            headless.reopen_capture_at_entry(data, "contended")
        except BaseException as exc:  # reported to the test's thread
            errors.append(exc)

    monkeypatch.setattr(headless, "PUBLICATION_LOCK_SEC", 0.05)
    monkeypatch.setattr(task_custody, "task_custody_lock", observed)
    other, entry = threading.Thread(target=publisher), threading.Thread(target=enter)
    other.start()
    try:
        assert holding.wait(10), "the other publisher never took the lock"
        entry.start()
        assert contended.wait(10), "the entry did not keep contending for the held lock"
        assert entry.is_alive(), "the entry returned while another publisher held the capture"
        held = load_task_result(data, "contended")
        assert (held["artifact_status"], held["artifacts"]) == (taken["artifact_status"], taken["artifacts"])
    finally:
        release.set()
        other.join(10)
        if entry.ident is not None:
            entry.join(10)
    assert not entry.is_alive() and not errors, errors
    assert _reopened(load_task_result(data, "contended"), manifest)


def test_an_entry_whose_first_read_fails_lets_the_row_decide_under_the_lock(tmp_path, monkeypatch):
    """#3248: the entry's first, lock-free read of the row fails. The start neither runs on that unknown nor fails
    there: the row the write reads under the custody lock decides, so the owed capture is still reopened, and a
    second entry over the reopened row changes nothing."""
    import errno

    from ouroboros import headless
    from ouroboros.task_results import load_task_result

    data, _taken, manifest = _entering(tmp_path, "unread")
    real_load, reads = headless.load_task_result, []

    def first_read_fails(root, task_id, **kwargs):
        reads.append(kwargs)
        if len(reads) == 1:
            raise OSError(errno.EIO, "the row was not read")
        return real_load(root, task_id, **kwargs)

    monkeypatch.setattr(headless, "load_task_result", first_read_fails)
    headless.reopen_capture_at_entry(data, "unread")
    assert reads[0] == {"strict": True}
    reopened = load_task_result(data, "unread")
    assert _reopened(reopened, manifest)
    reads.clear()
    headless.reopen_capture_at_entry(data, "unread")  # its first read fails again: the row owes nothing more
    assert reads[0] == {"strict": True} and load_task_result(data, "unread") == reopened


def test_a_queued_stop_after_the_entry_settles_missing_never_the_earlier_capture(tmp_path):
    """#3248's known residual: only the finish, a running task's Stop or a closure captures the entered tree. A Stop
    of the row queued again after its entry captures nothing, so the row settles ``missing``; the earlier capture
    stays history with its bytes and is never offered as the current one."""
    import pytest

    from ouroboros.headless import current_workspace_manifest, reopen_capture_at_entry
    from ouroboros.task_results import load_task_result
    from supervisor.cancel_publication import _cancel_result_fields

    data, _taken, manifest = _entering(tmp_path, "queued-stop")
    reopen_capture_at_entry(data, "queued-stop")
    entered = load_task_result(data, "queued-stop")
    write_task_result(data, "queued-stop", "cancelled", _cancel_observed=True, **_cancel_result_fields(
        {"id": "queued-stop", "workspace_root": entered["workspace_root"]}, existing=entered, result="Stopped."))
    row = load_task_result(data, "queued-stop")
    settled = (row["status"], row["artifact_status"], row["artifact_bundle"]["status"])
    assert settled == ("cancelled", "missing", "missing")
    assert manifest.is_file() and str(manifest) in {
        item["path"] for item in row["artifacts"] if item.get("kind") == "workspace_capture_history"}
    with pytest.raises(FileNotFoundError, match="current capture has not been published"):
        current_workspace_manifest(data, "queued-stop", row)


def test_task_artifact_endpoint_serves_only_declared_artifacts(tmp_path):
    data = tmp_path / "data"
    artifact_dir = task_artifacts_dir(data, "task-artifact")
    patch_path = artifact_dir / "workspace.patch"
    patch_path.write_bytes(b"diff --git a/a b/a\n")  # exact LF bytes on every platform
    write_task_result(
        data,
        "task-artifact",
        "completed",
        artifacts=[{"kind": "workspace_patch", "name": "workspace.patch", "path": str(patch_path), "size": patch_path.stat().st_size}],
        artifact_status="ready",
    )
    app = Starlette(routes=[Route("/api/tasks/{task_id}/artifacts/{name}", endpoint=api_task_artifact, methods=["GET"])])
    app.state.drive_root = data
    client = TestClient(app)

    _assert_file_response(client.get("/api/tasks/task-artifact/artifacts/workspace.patch"), b"diff --git a/a b/a\n")
    assert client.get("/api/tasks/task-artifact/artifacts/missing.patch").status_code == 404
    assert client.get("/api/tasks/task-artifact/artifacts/bad%5Cname").status_code == 400


def test_task_artifact_endpoint_serves_manifest_artifact_after_status_repair(tmp_path):
    from ouroboros.artifacts import copy_file_to_task_artifacts

    data = tmp_path / "data"
    source_dir = tmp_path / "Desktop"
    source_dir.mkdir()
    source = source_dir / "report.html"
    source.write_text("<h1>ok</h1>", encoding="utf-8")
    copy_file_to_task_artifacts(SimpleNamespace(drive_root=data, task_id="orphaned"), source, kind="user_file")
    write_task_result(
        data,
        "orphaned",
        "running",
        result_status="infra_failed",
        reason_code="provider_failure",
        result="provider failed before normal finalization",
    )
    (data / "state").mkdir(parents=True, exist_ok=True)
    (data / "state" / "queue_snapshot.json").write_text('{"pending": [], "running": []}', encoding="utf-8")
    app = Starlette(routes=[Route("/api/tasks/{task_id}/artifacts/{name}", endpoint=api_task_artifact, methods=["GET"])])
    app.state.drive_root = data

    response = TestClient(app).get("/api/tasks/orphaned/artifacts/report.html")

    _assert_file_response(response, b"<h1>ok</h1>")


def test_task_artifact_endpoint_serves_child_drive_artifact_read_only_after_status_repair(tmp_path):
    from ouroboros.artifacts import collect_task_artifact_records, copy_file_to_task_artifacts

    data = tmp_path / "data"
    # The task's OWN headless drive (host layout); a drive merely named by the row is no authority.
    child = data / "state" / "headless_tasks" / "childart" / "data"
    source_dir = tmp_path / "Desktop"
    source_dir.mkdir()
    source = source_dir / "report.html"
    source.write_text("<h1>child</h1>", encoding="utf-8")
    copy_file_to_task_artifacts(SimpleNamespace(drive_root=child, task_id="childart"), source, kind="user_file")
    child_artifacts = collect_task_artifact_records(child, "childart")
    write_task_result(
        child,
        "childart",
        "completed",
        result="done",
        artifacts=child_artifacts,
        artifact_status="ready",
        ts="2026-01-01T00:00:02Z",
    )
    write_task_result(
        data,
        "childart",
        "running",
        child_drive_root=str(child),
        workspace_root=str(tmp_path / "workspace"),
        result_status="infra_failed",
        reason_code="provider_failure",
        result="provider failed before normal finalization",
    )
    (data / "state").mkdir(parents=True, exist_ok=True)
    (data / "state" / "queue_snapshot.json").write_text('{"pending": [], "running": []}', encoding="utf-8")
    app = Starlette(routes=[Route("/api/tasks/{task_id}/artifacts/{name}", endpoint=api_task_artifact, methods=["GET"])])
    app.state.drive_root = data

    response = TestClient(app).get("/api/tasks/childart/artifacts/report.html")

    # The own child store is read when supported; neither outcome copies or creates it.
    _assert_file_response(response, b"<h1>child</h1>")
    assert not task_artifacts_dir(data, "childart", create=False).exists()


def test_task_artifact_endpoint_rejects_metadata_name_path_mismatch(tmp_path):
    data = tmp_path / "data"
    artifact_dir = task_artifacts_dir(data, "task-artifact")
    wrong_path = artifact_dir / "memory_export.json"
    wrong_path.write_text("{}", encoding="utf-8")
    write_task_result(
        data,
        "task-artifact",
        "completed",
        artifacts=[{"kind": "workspace_patch", "name": "workspace.patch", "path": str(wrong_path), "size": wrong_path.stat().st_size}],
        artifact_status="ready",
    )
    app = Starlette(routes=[Route("/api/tasks/{task_id}/artifacts/{name}", endpoint=api_task_artifact, methods=["GET"])])
    app.state.drive_root = data

    assert TestClient(app).get("/api/tasks/task-artifact/artifacts/workspace.patch").status_code == 500


def test_memory_export_includes_nested_memory_files(tmp_path):
    drive = tmp_path / "child"
    memory = drive / "memory"
    nested = memory / "knowledge" / "patterns"
    nested.mkdir(parents=True)
    (memory / "identity.md").write_text("id\n", encoding="utf-8")
    (nested / "cli.md").write_text("pattern\n", encoding="utf-8")

    export = build_memory_export(drive, {"id": "task-1", "memory_mode": "forked"})

    assert export["files"]["identity.md"] == "id\n"
    assert export["files"]["knowledge/patterns/cli.md"] == "pattern\n"


def test_startup_prune_removes_only_old_terminal_child_drives(tmp_path):
    data = tmp_path / "data"
    terminal_dir = data / "state" / "headless_tasks" / "oldterminal"
    pending_dir = data / "state" / "headless_tasks" / "oldpending"
    fresh_timestamp_dir = data / "state" / "headless_tasks" / "freshresult"
    terminal_drive = terminal_dir / "data"
    pending_drive = pending_dir / "data"
    fresh_timestamp_drive = fresh_timestamp_dir / "data"
    terminal_drive.mkdir(parents=True)
    pending_drive.mkdir(parents=True)
    fresh_timestamp_drive.mkdir(parents=True)

    now = time.time()
    old = now - (8 * 86400)
    old_iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(old))
    fresh_iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(now))
    write_task_result(data, "oldterminal", "completed", child_drive_root=str(terminal_drive), artifact_status="ready", result="done", ts=old_iso)
    write_task_result(data, "oldpending", "scheduled", child_drive_root=str(pending_drive), result="queued")
    write_task_result(data, "freshresult", "completed", child_drive_root=str(fresh_timestamp_drive), artifact_status="ready", result="done", ts=fresh_iso)
    os.utime(terminal_dir, (old, old))
    os.utime(pending_dir, (old, old))
    os.utime(fresh_timestamp_dir, (old, old))

    report = prune_headless_task_drives(data, retention_days=7, now=now, live=lambda _task: False)

    assert [item["task_id"] for item in report["pruned"]] == ["oldterminal"]
    assert not terminal_dir.exists()
    assert pending_dir.exists()
    assert fresh_timestamp_dir.exists()
    assert any(item["task_id"] == "oldpending" and item["reason"] == "parent_not_terminal" for item in report["skipped"])
    assert any(item["task_id"] == "freshresult" and item["reason"] == "younger_than_retention" for item in report["skipped"])


def test_prune_settles_only_durably_terminal_rows_never_a_projection(tmp_path):
    data = tmp_path / "data"
    task_drive = data / "task_drives" / "stalerun"
    child_dir = data / "state" / "headless_tasks" / "stalechild"
    child_drive = child_dir / "data"
    task_drive.mkdir(parents=True)
    child_drive.mkdir(parents=True)
    (task_drive / "scratch.txt").write_text("scratch", encoding="utf-8")
    (child_drive / "scratch.txt").write_text("child", encoding="utf-8")
    (data / "state").mkdir(parents=True, exist_ok=True)
    (data / "state" / "queue_snapshot.json").write_text('{"pending": [], "running": []}', encoding="utf-8")

    now = time.time()
    old = now - (8 * 86400)
    old_iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(old))
    for task_id, extra in (
        ("stalerun", {}),
        ("stalechild", {"child_drive_root": str(child_drive)}),
    ):
        write_task_result(
            data,
            task_id,
            "running",
            result_status="infra_failed",
            reason_code="provider_failure",
            result="provider failed",
            ts=old_iso,
            **extra,
        )
    os.utime(task_drive, (old, old))
    os.utime(child_dir, (old, old))

    # A projection is not custody: only the DURABLE row settles a drive (TZ-1 A).
    direct_report = prune_task_drives(data, retention_days=7, now=now, live=lambda _task: False)
    child_report = prune_headless_task_drives(data, retention_days=7, now=now, live=lambda _task: False)
    assert [item["reason"] for item in direct_report["skipped"]] == ["task_not_terminal"]
    assert [item["reason"] for item in child_report["skipped"]] == ["parent_not_terminal"]
    assert task_drive.exists() and child_dir.exists()

    from ouroboros.task_status import reconcile_orphaned_running_tasks

    assert reconcile_orphaned_running_tasks(data) == 2  # the reconciler persists what the projection says
    direct_report = prune_task_drives(data, retention_days=7, now=now, live=lambda _task: False)
    child_report = prune_headless_task_drives(data, retention_days=7, now=now, live=lambda _task: False)

    assert [item["task_id"] for item in direct_report["pruned"]] == ["stalerun"]
    assert [item["task_id"] for item in child_report["pruned"]] == ["stalechild"]
    assert not task_drive.exists()
    assert not child_dir.exists()


def test_startup_prune_removes_only_old_terminal_task_scratch(tmp_path):
    data = tmp_path / "data"
    old_terminal = data / "task_drives" / "oldterminal"
    old_pending = data / "task_drives" / "oldpending"
    fresh_terminal = data / "task_drives" / "freshterminal"
    for path in (old_terminal, old_pending, fresh_terminal):
        path.mkdir(parents=True)
        (path / "scratch.txt").write_text("scratch", encoding="utf-8")

    now = time.time()
    old = now - (8 * 86400)
    old_iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(old))
    fresh_iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(now))
    write_task_result(data, "oldterminal", "completed", result="done", ts=old_iso)
    write_task_result(data, "oldpending", "running", result="running")
    write_task_result(data, "freshterminal", "completed", result="done", ts=fresh_iso)
    os.utime(old_terminal, (old, old))
    os.utime(old_pending, (old, old))
    os.utime(fresh_terminal, (old, old))

    report = prune_task_drives(data, retention_days=7, now=now, live=lambda _task: False)

    assert [item["task_id"] for item in report["pruned"]] == ["oldterminal"]
    assert not old_terminal.exists()
    assert old_pending.exists()
    assert fresh_terminal.exists()
    assert any(item["task_id"] == "oldpending" and item["reason"] == "task_not_terminal" for item in report["skipped"])
    assert any(item["task_id"] == "freshterminal" and item["reason"] == "younger_than_retention" for item in report["skipped"])


def test_external_child_task_budget_uses_parent_drive_state(tmp_path, monkeypatch):
    from ouroboros import usage_accounting
    from ouroboros.agent import Env, OuroborosAgent

    repo = tmp_path / "repo"
    parent = tmp_path / "parent-data"
    child = tmp_path / "child-data"
    for root in (repo, parent, child):
        root.mkdir()
    for drive in (parent, child):
        (drive / "state").mkdir()
        (drive / "logs").mkdir()
    # Compatibility projections are deliberately misleading here: the physical-attempt
    # ledger in the parent budget root is the sole monetary authority.
    (parent / "state" / "state.json").write_text('{"spent_usd": 0.0}\n', encoding="utf-8")
    (child / "state" / "state.json").write_text('{"spent_usd": 0.0}\n', encoding="utf-8")
    reservation = usage_accounting.reserve_attempt(usage_accounting.AttemptRequest(
        model="test/model",
        provider="test",
        reservation_usd=9.0,
        drive_root=parent,
        task_id="prior-task",
        root_task_id="prior-task",
        source="test",
    ))
    usage_accounting.mark_dispatched(reservation)
    usage_accounting.settle_attempt(reservation, {}, cost_usd=9.0, cost_final=True)

    monkeypatch.setenv("TOTAL_BUDGET", "10")
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    monkeypatch.setattr("ouroboros.agent.build_llm_messages", lambda **kwargs: ([], {}))

    agent = OuroborosAgent(Env(repo_dir=repo, drive_root=child))
    ctx, _messages, cap_info = agent._prepare_task_context({
        "id": "budget-task",
        "type": "task",
        "text": "x",
        "budget_drive_root": str(parent),
    })

    assert cap_info["budget_remaining"] == 1.0
    assert ctx.task_metadata["budget_drive_root"] == str(parent)


def test_task_artifact_endpoint_serves_exact_chat_media_without_task_result(tmp_path):
    from ouroboros.artifacts import collect_task_artifact_records, store_chat_media_bytes

    data = tmp_path / "data"
    stored = store_chat_media_bytes(data, "ephemeral1", b"photo-bytes", "image/png")
    assert stored is not None
    app = Starlette(routes=[Route("/api/tasks/{task_id}/artifacts/{name}", endpoint=api_task_artifact, methods=["GET"])])
    app.state.drive_root = data
    client = TestClient(app)

    response = client.get(f"/api/tasks/ephemeral1/artifacts/{stored['name']}")
    _assert_file_response(response, b"photo-bytes")
    assert collect_task_artifact_records(data, "ephemeral1") == []

    assert client.get("/api/tasks/ephemeral1/artifacts/chat-media-bad.png").status_code == 404
