"""A review run's bookkeeping never becomes a task deliverable.

A task whose trace carries a review run still ends with an artifact bundle of
its own user deliverables only: the artifact manifest, its lock and the
verification receipts are never served as task artifacts, at the terminal
pipeline and after a child's result is copied back. The applied-source download
and the ``review_reference`` row these tests once also checked were written by
the task-acceptance checkpoint publisher, which is gone; old stored panels that
carry an applied source still render through the review projection's readers.
"""

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import agent_task_pipeline, artifacts
from ouroboros.agent import Env
from ouroboros.gateway.tasks import api_task_artifact
from ouroboros.headless import copy_child_task_result
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.task_status import load_effective_task_result
from tests._store_shared import remove_store_tree

pytestmark = pytest.mark.serial


def _run(task_attempt=1):
    """One applied host review run with a long finding list (a recorded panel's shape)."""
    findings = [{"id": f"f{i}", "severity": "low", "item": f"full finding {i}",
                 "evidence": "complete context", "recommendation": "consider later"} for i in range(80)]
    return {"request": {"surface": "task_acceptance", "task_id": "applied"},
            "panel_id": "panel_exact", "authority": "host_root", "task_attempt": task_attempt, "aggregate_signal": "PASS",
            "actors": [{"slot_id": "s1", "signal": "PASS", "status": "ok",
                        "parsed": {"verdict": "PASS", "outcome_tier": "solved", "findings": findings},
                        "criteria_refs_unresolved": [{"criterion": "complete", "supported_evidence_resolves": True}]}],
            "parsed_findings": findings, "enforcement_impact": "allows_completion",
            "dialogue": {"status": "inconclusive"}}


def _finish(root, *, canonical=None, chat_id=0, user_file=False):
    root.mkdir(parents=True, exist_ok=True)
    repo = root / "repo"
    repo.mkdir(exist_ok=True)
    if user_file:
        artifacts.store_task_artifact_bytes(root, "applied", "notes.lock", b"user deliverable", kind="user_file")
    env = Env(repo_dir=repo, drive_root=root, budget_drive_root=canonical)
    task = {"id": "applied", "type": "task", "chat_id": chat_id, "text": "Give an answer",
            "_skip_post_task_synthesis": True}
    if canonical:
        task["budget_drive_root"] = str(canonical)
    trace = {"tool_calls": [], "review_runs": [_run()]}
    pending = []
    agent_task_pipeline.emit_task_results(
        env, None, None, pending, task, "The requested answer.", {"rounds": 1}, trace,
        start_time=0.0, drive_logs=root / "logs",
    )
    stored = load_task_result(root, "applied")
    assert stored["status"] == "completed"
    return task, stored, pending


def _assert_no_bookkeeping_downloads(root):
    """The manifest, its lock and the verification receipts are never served as task artifacts."""
    app = Starlette(routes=[Route("/api/tasks/{task_id}/artifacts/{name}", api_task_artifact)])
    app.state.drive_root = root
    with TestClient(app) as client:
        for name in (artifacts._ARTIFACT_MANIFEST, artifacts._ARTIFACT_MANIFEST + ".lock", "verification_receipts.jsonl"):
            assert client.get(f"/api/tasks/applied/artifacts/{name}").status_code == 404


@pytest.mark.parametrize("chat_id", [0, 1])
@pytest.mark.parametrize("user_file", [False, True])
def test_terminal_pipeline_keeps_review_bookkeeping_out_of_the_deliverables(tmp_path, monkeypatch, chat_id, user_file):
    monkeypatch.setattr(agent_task_pipeline, "_run_post_task_processing_async", lambda *a, **k: None)
    _task, stored, pending = _finish(tmp_path, chat_id=chat_id, user_file=user_file)

    expected_artifacts = "ready" if user_file else "not_applicable"
    assert stored["artifact_bundle"]["status"] == expected_artifacts
    assert stored["outcome_axes"]["artifacts"]["status"] == expected_artifacts
    assert [row["name"] for row in stored["artifacts"]] == (["notes.lock"] if user_file else [])
    done = next(row for row in pending if row.get("type") == "task_done")
    assert done["artifact_status"] == expected_artifacts
    assert [row["name"] for row in done["artifact_bundle"]["artifacts"]] == (["notes.lock"] if user_file else [])
    _assert_no_bookkeeping_downloads(tmp_path)
    for materialize in (False, True):
        effective = load_effective_task_result(tmp_path, "applied", materialize_artifacts=materialize)
        assert effective["artifacts"] == stored["artifacts"]
        assert effective["artifact_bundle"]["status"] == expected_artifacts


@pytest.mark.parametrize("user_file", [False, True])
def test_child_copyback_keeps_the_review_projection_and_only_user_deliverables(tmp_path, monkeypatch, user_file):
    canonical, child = tmp_path / "canonical", tmp_path / "child"
    canonical.mkdir()
    monkeypatch.setattr(agent_task_pipeline, "_run_post_task_processing_async", lambda *a, **k: None)
    write_task_result(canonical, "applied", "running", child_drive_root=str(child))
    _, child_result, _ = _finish(child, canonical=canonical, user_file=user_file)
    copied = copy_child_task_result(canonical, {"id": "applied", "drive_root": str(child)})
    remove_store_tree(child)  # the child drive's store stays open in this process

    expected = "ready" if user_file else "not_applicable"
    assert copied["artifact_bundle"]["status"] == expected
    assert [row["name"] for row in copied["artifacts"]] == (["notes.lock"] if user_file else [])
    assert all(row["kind"] == "user_file" for row in copied["artifacts"])
    assert copied["review_projection"] == child_result["review_projection"]
    final = load_effective_task_result(canonical, "applied")
    assert final["artifact_bundle"]["status"] == expected
    _assert_no_bookkeeping_downloads(canonical)
    if user_file:
        assert artifacts.task_artifact_dir_path(canonical, "applied").joinpath("notes.lock").read_bytes() == b"user deliverable"


@pytest.mark.parametrize("status", ["pending", "finalizing", "failed", "partial", "missing"])
def test_present_but_incomplete_user_record_does_not_upgrade_empty_bundle(tmp_path, status):
    from ouroboros.outcomes import artifact_bundle_from_result

    path = tmp_path / "partial.txt"
    path.write_bytes(b"some bytes exist")
    record = {**artifacts.artifact_record(path, kind="user_file"), "status": status}
    bundle = artifact_bundle_from_result({"artifacts": [record], "artifact_status": "not_applicable"})
    assert bundle["status"] != "ready"
    assert bundle["artifacts"][0]["status"] == status


@pytest.mark.parametrize("status", ["pending", "finalizing", "failed", "ready_no_changes"])
def test_ready_user_record_preserves_independent_bundle_state(tmp_path, status):
    from ouroboros.outcomes import artifact_bundle_from_result

    path = tmp_path / "answer.txt"
    path.write_bytes(b"real result")
    bundle = artifact_bundle_from_result({
        "artifacts": [artifacts.artifact_record(path, kind="user_file")], "artifact_status": status,
    })
    assert bundle["status"] == status
