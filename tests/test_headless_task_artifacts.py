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
    refreshed_ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert refreshed_ledger["outcome_axes"]["artifacts"]["status"] == ARTIFACT_STATUS_READY_WITH_CHANGES

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


def test_task_artifact_endpoint_serves_only_declared_artifacts(tmp_path):
    data = tmp_path / "data"
    artifact_dir = task_artifacts_dir(data, "task-artifact")
    patch_path = artifact_dir / "workspace.patch"
    patch_path.write_text("diff --git a/a b/a\n", encoding="utf-8")
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

    assert client.get("/api/tasks/task-artifact/artifacts/workspace.patch").text.startswith("diff --git")
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

    assert response.status_code == 200
    assert response.text == "<h1>ok</h1>"


def test_task_artifact_endpoint_rebases_child_drive_artifact_after_status_repair(tmp_path):
    from ouroboros.artifacts import collect_task_artifact_records, copy_file_to_task_artifacts

    data = tmp_path / "data"
    child = tmp_path / "child"
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

    parent_artifact = task_artifacts_dir(data, "childart", create=False) / "report.html"
    assert response.status_code == 200
    assert response.text == "<h1>child</h1>"
    assert parent_artifact.read_text(encoding="utf-8") == "<h1>child</h1>"


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

    report = prune_headless_task_drives(data, retention_days=7, now=now)

    assert [item["task_id"] for item in report["pruned"]] == ["oldterminal"]
    assert not terminal_dir.exists()
    assert pending_dir.exists()
    assert fresh_timestamp_dir.exists()
    assert any(item["task_id"] == "oldpending" and item["reason"] == "parent_not_terminal" for item in report["skipped"])
    assert any(item["task_id"] == "freshresult" and item["reason"] == "younger_than_retention" for item in report["skipped"])


def test_startup_prune_uses_effective_terminal_status(tmp_path):
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

    direct_report = prune_task_drives(data, retention_days=7, now=now)
    child_report = prune_headless_task_drives(data, retention_days=7, now=now)

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

    report = prune_task_drives(data, retention_days=7, now=now)

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
    assert response.status_code == 200
    assert response.content == b"photo-bytes"
    assert collect_task_artifact_records(data, "ephemeral1") == []

    assert client.get("/api/tasks/ephemeral1/artifacts/chat-media-bad.png").status_code == 404
