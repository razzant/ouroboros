"""materialize_artifacts contract tests (v6.90.x P2).

`effective_task_result(..., materialize_artifacts=False)` is a "status/cost
projection only" read for hot display surfaces (history annotation,
api_tasks_list, the SSE follow loop, api_logs_tail discovery): it must skip the
entire artifact block — including the MUTATING child-artifact rebase and the
collect_task_artifact_records file scans — and the task-tree disposition hash
lookup, and must never carry sha-bearing/disposition claims. Every sha-economy
consumer keeps the True default, so a terminal child's `_child_result_sha256`
is identical no matter how many False-path reads happened in between. The
orphan reconciler decides on a False read and materializes only the row it
heals, so a live child's scratch tree is never promoted by the periodic sweep
(issue #1230).
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from ouroboros.task_results import write_task_result
from ouroboros.task_status import find_child_tasks, load_effective_task_result


def _seed_child_drive_scenario(tmp_path):
    """Parent result referencing a terminal child drive that holds an artifact."""
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
        "completed",
        result="done",
        child_drive_root=str(child),
        # Legacy mirrored disposition fields on disk must be stripped either way.
        child_result_disposition="integrated",
        child_result_disposition_sha256="deadbeef",
        delegation_role="subagent",
        parent_task_id="parent1",
        root_task_id="parent1",
    )
    (data / "state").mkdir(parents=True, exist_ok=True)
    (data / "state" / "queue_snapshot.json").write_text('{"pending": [], "running": []}', encoding="utf-8")
    return data, child


def test_false_read_skips_artifact_block_and_disposition(tmp_path, monkeypatch):
    data, child = _seed_child_drive_scenario(tmp_path)
    import ouroboros.artifacts as artifacts_mod

    calls = {"collect": 0, "copy": 0}
    real_collect = artifacts_mod.collect_task_artifact_records
    real_copy = artifacts_mod.copy_file_to_task_artifacts
    monkeypatch.setattr(
        artifacts_mod, "collect_task_artifact_records",
        lambda *a, **k: calls.__setitem__("collect", calls["collect"] + 1) or real_collect(*a, **k),
    )
    monkeypatch.setattr(
        artifacts_mod, "copy_file_to_task_artifacts",
        lambda *a, **k: calls.__setitem__("copy", calls["copy"] + 1) or real_copy(*a, **k),
    )

    row = load_effective_task_result(data, "childart", materialize_artifacts=False)

    assert row["status"] == "completed"
    assert row["result"] == "done"
    # No artifact machinery ran and no files were copied to the parent.
    assert calls == {"collect": 0, "copy": 0}
    assert not (data / "task_results" / "artifacts" / "childart" / "report.html").exists()
    # A False row never carries sha-bearing/disposition claims — even the legacy
    # mirrored on-disk fields are stripped.
    for field in (
        "child_result_disposition",
        "child_result_disposition_sha256",
        "child_result_disposition_reason",
        "child_result_disposition_source",
        "parent_decision_child_result_sha256",
        "terminal_child_result_snapshot",
    ):
        assert field not in row


def test_child_result_sha_stable_across_false_path_reads(tmp_path):
    from ouroboros.tools.join_ledger import _child_result_sha256

    data, child = _seed_child_drive_scenario(tmp_path)

    sha_before = _child_result_sha256(load_effective_task_result(data, "childart"))
    # Any number of projection-only reads in between must not perturb the sha
    # economy (they perform no writes at all).
    for _ in range(3):
        load_effective_task_result(data, "childart", materialize_artifacts=False)
        find_child_tasks(data, parent_task_id="parent1", root_task_id="parent1", materialize_artifacts=False)
    sha_after = _child_result_sha256(load_effective_task_result(data, "childart"))

    assert sha_before == sha_after


def test_true_default_still_materializes_child_artifacts(tmp_path):
    """The default path keeps the read-repair durability: child artifacts are
    rebased onto the parent drive (api_task_artifact depends on it)."""
    data, child = _seed_child_drive_scenario(tmp_path)

    row = load_effective_task_result(data, "childart")

    rebased = data / "task_results" / "artifacts" / "childart" / "report.html"
    assert rebased.exists()
    assert rebased.read_text(encoding="utf-8") == "<h1>child</h1>"
    assert any(
        (item.get("name") or "") == "report.html" for item in row.get("artifacts") or []
    )


def test_generic_receipt_named_output_never_owns_receipt_authority(tmp_path):
    """A user/process output may keep its name, never the receipt SSOT path."""
    from ouroboros.artifacts import (
        collect_task_artifact_records,
        copy_file_to_task_artifacts,
    )
    from ouroboros.outcomes import (
        append_verification_receipt,
        read_verification_receipts,
        verification_receipts_path,
    )

    drive = tmp_path / "data"
    source = tmp_path / "outside" / "verification_receipts.jsonl"
    source.parent.mkdir()
    source.write_text('{"artifact":"v1"}\n', encoding="utf-8")
    ctx = SimpleNamespace(drive_root=drive, task_id="receipt-collision")

    first = copy_file_to_task_artifacts(ctx, source, kind="process_output")
    authority = verification_receipts_path(drive, ctx.task_id)
    assert first is not None
    assert first["name"].startswith("verification_receipts.")
    assert first["name"].endswith(".jsonl")
    assert first["path"] != str(authority)

    assert append_verification_receipt(
        drive,
        ctx.task_id,
        {"status": "pass", "criterion_id": "host"},
    )
    source.write_text('{"artifact":"v2"}\n', encoding="utf-8")
    second = copy_file_to_task_artifacts(ctx, source, kind="process_output")

    assert second is not None
    assert second["path"] == first["path"]
    assert any(
        row.get("criterion_id") == "host"
        for row in read_verification_receipts(drive, ctx.task_id)
    )
    records = collect_task_artifact_records(drive, ctx.task_id)
    assert [item["path"] for item in records] == [first["path"]]

    from ouroboros.tools.core import _write_file
    from ouroboros.tools.registry import ToolContext

    repo = tmp_path / "repo"
    repo.mkdir()
    tool_ctx = ToolContext(repo_dir=repo, drive_root=drive, task_id=ctx.task_id)
    blocked = _write_file(
        tool_ctx,
        path="verification_receipts.jsonl",
        content='{"generic":"overwrite"}\n',
        root="artifact_store",
    )
    assert "verification receipt authority path is reserved" in blocked
    nested = _write_file(
        tool_ctx,
        path="nested/verification_receipts.jsonl",
        content='{"generic":"allowed"}\n',
        root="artifact_store",
    )
    assert nested.startswith("OK: wrote artifact_store:nested/")


def test_history_and_tasks_list_paths_do_no_artifact_work(tmp_path, monkeypatch):
    """Counter-assert: GET /api/chat/history and GET /api/tasks perform ZERO
    collect_task_artifact_records / copy_file_to_task_artifacts calls."""
    import ouroboros.artifacts as artifacts_mod
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.gateway.tasks import api_tasks_list

    data, child = _seed_child_drive_scenario(tmp_path)
    logs = data / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "chat.jsonl").write_text("", encoding="utf-8")
    (logs / "progress.jsonl").write_text(
        json.dumps({"ts": "2026-01-01T00:00:01Z", "content": "step", "task_id": "childart"}) + "\n",
        encoding="utf-8",
    )

    calls = {"collect": 0, "copy": 0}
    monkeypatch.setattr(
        artifacts_mod, "collect_task_artifact_records",
        lambda *a, **k: calls.__setitem__("collect", calls["collect"] + 1) or [],
    )
    monkeypatch.setattr(
        artifacts_mod, "copy_file_to_task_artifacts",
        lambda *a, **k: calls.__setitem__("copy", calls["copy"] + 1) or None,
    )

    history = make_chat_history_endpoint(data)
    response = asyncio.run(history(SimpleNamespace(query_params={"limit": "10"})))
    payload = json.loads(response.body.decode("utf-8"))["messages"]
    assert any(item.get("task_id") == "childart" for item in payload)

    request = SimpleNamespace(
        query_params={},
        app=SimpleNamespace(state=SimpleNamespace(drive_root=data)),
    )
    response = asyncio.run(api_tasks_list(request))
    tasks = json.loads(response.body.decode("utf-8"))["tasks"]
    assert any(task.get("task_id") == "childart" for task in tasks)

    assert calls == {"collect": 0, "copy": 0}


def _iso(epoch: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _spy_effective_reads(monkeypatch) -> list:
    """Record the ``materialize_artifacts`` flag of every reconciler read (real calls)."""
    from ouroboros import task_status

    flags: list = []
    real = task_status.load_effective_task_result

    def _spy(drive_root, tid, materialize_artifacts=True):
        flags.append(bool(materialize_artifacts))
        return real(drive_root, tid, materialize_artifacts=materialize_artifacts)

    monkeypatch.setattr(task_status, "load_effective_task_result", _spy)
    return flags


def _count_copies(monkeypatch) -> dict:
    import ouroboros.artifacts as artifacts_mod

    calls = {"copy": 0}
    real_copy = artifacts_mod.copy_file_to_task_artifacts
    monkeypatch.setattr(
        artifacts_mod, "copy_file_to_task_artifacts",
        lambda *a, **k: calls.__setitem__("copy", calls["copy"] + 1) or real_copy(*a, **k),
    )
    return calls


def test_reconcile_skips_a_live_running_child_without_materializing(tmp_path, monkeypatch):
    """A RUNNING child with a large scratch tree under its artifact dir costs the
    300-s sweep one status projection and zero artifact transfers."""
    import time

    from ouroboros.artifacts import _ARTIFACT_MANIFEST, copy_file_to_task_artifacts, task_artifact_dir_path
    from ouroboros.task_status import reconcile_orphaned_running_tasks

    now = 1_800_000_000.0
    monkeypatch.setattr(time, "time", lambda: now)
    data, child, tid = tmp_path / "data", tmp_path / "child", "livechild"
    source = tmp_path / "report.html"
    source.write_text("<h1>live</h1>", encoding="utf-8")
    copy_file_to_task_artifacts(SimpleNamespace(drive_root=child, task_id=tid), source, kind="user_file")
    sandbox = task_artifact_dir_path(child, tid) / "verification-sandbox"
    for index in range(300):
        target = sandbox / (".git" if index % 2 else "venv") / f"file{index}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"scratch {index}\n", encoding="utf-8")
    write_task_result(child, tid, "running", result="working", ts=_iso(now - 600))
    write_task_result(data, tid, "running", result="working", child_drive_root=str(child), ts=_iso(now - 600))
    (data / "state").mkdir(parents=True, exist_ok=True)
    (data / "state" / "queue_snapshot.json").write_text(
        json.dumps({"ts": _iso(now), "pending": [], "running": [{"id": tid, "task": {"id": tid}}]}),
        encoding="utf-8",
    )
    child_manifest = task_artifact_dir_path(child, tid) / _ARTIFACT_MANIFEST
    manifest_before = child_manifest.read_bytes()
    flags, copies = _spy_effective_reads(monkeypatch), _count_copies(monkeypatch)

    assert reconcile_orphaned_running_tasks(data) == 0

    assert flags == [False]
    assert copies == {"copy": 0}
    assert not (data / "task_results" / "artifacts" / tid).exists()
    assert child_manifest.read_bytes() == manifest_before
    assert json.loads((data / "task_results" / f"{tid}.json").read_text(encoding="utf-8"))["status"] == "running"


def test_reconcile_heals_a_genuine_orphan_with_full_artifact_custody(tmp_path, monkeypatch):
    """The positive path: a parent row stuck at ``running`` over a finished child
    drive is settled from the materializing read, so the persisted row carries
    the promoted artifact, its bundle and the terminal quiz settlement."""
    import time

    from ouroboros import owner_quiz
    from ouroboros.artifacts import collect_task_artifact_records, copy_file_to_task_artifacts
    from ouroboros.task_status import SETTLED_STATUSES, reconcile_orphaned_running_tasks
    from ouroboros.utils import append_jsonl

    now = 1_800_000_000.0
    monkeypatch.setattr(time, "time", lambda: now)
    data, child, tid = tmp_path / "data", tmp_path / "child", "orphanchild"
    source = tmp_path / "report.html"
    source.write_text("<h1>child</h1>", encoding="utf-8")
    copy_file_to_task_artifacts(SimpleNamespace(drive_root=child, task_id=tid), source, kind="user_file")
    write_task_result(child, tid, "completed", result="done", artifacts=collect_task_artifact_records(child, tid),
                      artifact_status="ready", ts=_iso(now - 3_600))
    write_task_result(data, tid, "running", result="Task is running.", child_drive_root=str(child),
                      ts=_iso(now - 7_200))
    owner_quiz.record_asked(data, tid, quiz_id="q1", question="Continue?", options=["yes", "no"])
    (data / "state").mkdir(parents=True, exist_ok=True)
    (data / "state" / "queue_snapshot.json").write_text(
        json.dumps({"ts": _iso(now), "pending": [], "running": []}), encoding="utf-8",
    )
    events = data / "logs" / "events.jsonl"
    append_jsonl(events, {"ts": _iso(now - 7_000), "type": "llm_round", "task_id": tid})
    append_jsonl(events, {"ts": _iso(now - 6_000), "type": "worker_boot"})
    flags, copies = _spy_effective_reads(monkeypatch), _count_copies(monkeypatch)
    expired: list = []

    assert reconcile_orphaned_running_tasks(data, expired_quizzes=expired) == 1

    assert flags == [False, True]
    assert copies["copy"] >= 1
    # The persisted bytes and the promoted file are checked BEFORE any materializing
    # loader runs again, so the oracle cannot repair what the sweep left undone.
    on_disk = json.loads((data / "task_results" / f"{tid}.json").read_text(encoding="utf-8"))
    assert on_disk["status"] in SETTLED_STATUSES
    assert on_disk["artifact_status"] and isinstance(on_disk.get("artifact_bundle"), dict)
    promoted = data / "task_results" / "artifacts" / tid / "report.html"
    assert promoted.read_text(encoding="utf-8") == "<h1>child</h1>"
    assert expired == [(tid, "q1")]
    assert owner_quiz.quiz_states(data, tid)["q1"]["state"] == "expired_terminal"
    direct = load_effective_task_result(data, tid)
    for key in ("status", "artifact_status", "artifact_bundle"):
        assert on_disk[key] == direct[key], key
    assert on_disk.get("status_reconciled_from") == direct.get("status_reconciled_from")
