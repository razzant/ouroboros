"""Applied review sources retain download custody without becoming task deliverables."""

import copy
import hashlib
import shutil
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import agent_task_pipeline, artifacts, utils
from ouroboros.agent import Env
from ouroboros.gateway.history import _collect_progress_rows, _make_thread_filter
from ouroboros.gateway.tasks import api_task_artifact
from ouroboros.headless import copy_child_task_result
from ouroboros.projects_registry import bind_task_to_project, create_project
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.task_status import load_effective_task_result
from supervisor.events_worker_reports import _handle_log_event
from supervisor.worker_process import WORKER_LOG_SINK_SUPPRESSED_TYPES
from tests.test_acceptance_publication import _run

pytestmark = pytest.mark.serial


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


def _download(root, stored):
    panel = stored["review_projection"]["panels"][0]
    ref = panel["applied_source_ref"]
    app = Starlette(routes=[Route("/api/tasks/{task_id}/artifacts/{name}", api_task_artifact)])
    app.state.drive_root = root
    with TestClient(app) as client:
        response = client.get(f"/api/tasks/applied/artifacts/{ref['path']}")
        assert response.status_code == 200
        assert hashlib.sha256(response.content).hexdigest() == ref["sha256"]
        assert len(response.content) == ref["bytes"]
        assert len(response.json()["actors"][0]["parsed"]["findings"]) == 80
        for name in (artifacts._ARTIFACT_MANIFEST, artifacts._ARTIFACT_MANIFEST + ".lock", "verification_receipts.jsonl"):
            assert client.get(f"/api/tasks/applied/artifacts/{name}").status_code == 404
    return ref


@pytest.mark.parametrize("chat_id,project_bound", [(0, False), (1, False), (0, True), (1, True)])
@pytest.mark.parametrize("user_file", [False, True])
def test_terminal_pipeline_routes_review_and_downloads_without_bookkeeping_deliverable(
    tmp_path, monkeypatch, chat_id, project_bound, user_file,
):
    captured = []
    monkeypatch.setattr(utils, "_log_sink", lambda row: captured.append(dict(row)))
    monkeypatch.setattr(agent_task_pipeline, "_run_post_task_processing_async", lambda *a, **k: None)
    project_chat = 0
    if project_bound:
        project_chat = create_project(tmp_path, "review-project")["chat_id"]
        bind_task_to_project(tmp_path, "applied", "review-project", origin={"absent": "system"})
    task, stored, pending = _finish(tmp_path, chat_id=chat_id, user_file=user_file)

    expected_artifacts = "ready" if user_file else "not_applicable"
    assert stored["artifact_bundle"]["status"] == expected_artifacts
    assert stored["outcome_axes"]["artifacts"]["status"] == expected_artifacts
    assert [row["name"] for row in stored["artifacts"]] == (["notes.lock"] if user_file else [])
    done = next(row for row in pending if row.get("type") == "task_done")
    assert done["artifact_status"] == expected_artifacts
    assert not any(artifacts.is_task_bookkeeping_artifact(row) for row in done["artifact_bundle"]["artifacts"])
    _download(tmp_path, stored)
    for materialize in (False, True):
        effective = load_effective_task_result(tmp_path, "applied", materialize_artifacts=materialize)
        assert effective["artifacts"] == stored["artifacts"]
        assert effective["artifact_bundle"]["status"] == expected_artifacts

    reference = next(row for row in captured if row.get("type") == "review_reference")
    assert reference["chat_id"] == chat_id  # Env intentionally carries no current_chat_id.
    assert reference["type"] not in WORKER_LOG_SINK_SUPPRESSED_TYPES
    live = []
    supervisor = SimpleNamespace(RUNNING={"applied": {"task": task}}, DRIVE_ROOT=tmp_path,
                                 bridge=SimpleNamespace(push_log=live.append))
    _handle_log_event({"type": "log_event", "data": reference}, supervisor)
    assert live[0]["chat_id"] == (project_chat or chat_id)
    bindings = {"applied": project_chat} if project_bound else {}
    chats = {project_chat} if project_bound else set()
    main_rows, _ = _collect_progress_rows(
        tmp_path / "logs" / "progress.jsonl", tmp_path / "logs" / "archive", 20,
        _make_thread_filter(1, chats, [], bindings),
    )
    main_references = [row for row in main_rows if row.get("system_type") == "review_reference"]
    assert bool(main_references) is (chat_id == 1 and not project_bound)
    if project_bound:
        project_rows, _ = _collect_progress_rows(
            tmp_path / "logs" / "progress.jsonl", tmp_path / "logs" / "archive", 20,
            _make_thread_filter(project_chat, chats, [], bindings),
        )
        assert any(row.get("system_type") == "review_reference" for row in project_rows)


@pytest.mark.parametrize("user_file", [False, True])
def test_canonical_source_survives_actual_child_copyback_and_cleanup(tmp_path, monkeypatch, user_file):
    canonical, child = tmp_path / "canonical", tmp_path / "child"
    canonical.mkdir()
    monkeypatch.setattr(agent_task_pipeline, "_run_post_task_processing_async", lambda *a, **k: None)
    write_task_result(canonical, "applied", "running", child_drive_root=str(child))
    _, child_result, _ = _finish(child, canonical=canonical, user_file=user_file)
    source = _download(canonical, load_effective_task_result(canonical, "applied"))
    assert not (artifacts.task_artifact_dir_path(child, "applied") / source["path"]).exists()
    copied = copy_child_task_result(canonical, {"id": "applied", "drive_root": str(child)})
    shutil.rmtree(child)

    expected = "ready" if user_file else "not_applicable"
    assert copied["artifact_bundle"]["status"] == expected
    assert [row["name"] for row in copied["artifacts"]] == (["notes.lock"] if user_file else [])
    assert not any(artifacts.is_task_bookkeeping_artifact(row) for row in copied["artifacts"])
    assert copied["review_projection"] == child_result["review_projection"]
    final = load_effective_task_result(canonical, "applied")
    assert final["artifact_bundle"]["status"] == expected
    _download(canonical, final)
    if user_file:
        assert artifacts.task_artifact_dir_path(canonical, "applied").joinpath("notes.lock").read_bytes() == b"user deliverable"


@pytest.mark.parametrize("status", ["ready", "pending", "finalizing", "failed", "ready_no_changes"])
def test_existing_bookkeeping_projection_preserves_independent_states(status):
    review = {"name": "acceptance.json", "kind": "task_acceptance_review", "status": "ready"}
    row = {"artifacts": [review], "artifact_status": status,
           "artifact_bundle": {"artifacts": [review], "status": status},
           "outcome_axes": {"artifacts": {"status": status}}}
    before = copy.deepcopy(row)
    projected = artifacts.project_deliverable_artifacts(row)
    expected = "not_applicable" if status == "ready" else status
    assert projected["artifacts"] == [] and projected["artifact_bundle"]["artifacts"] == []
    assert projected["artifact_status"] == projected["artifact_bundle"]["status"] == expected
    assert projected["outcome_axes"]["artifacts"]["status"] == expected
    assert row == before


def test_stale_review_only_replica_does_not_restore_ready_status(tmp_path):
    canonical, child = tmp_path / "canonical", tmp_path / "child"
    source = artifacts.store_task_artifact_bytes(canonical, "applied", "review.json", b"canonical source", kind="task_acceptance_review")
    review = artifacts.artifact_record(artifacts.task_artifact_dir_path(canonical, "applied") / source["path"], kind="task_acceptance_review")
    fields = {"artifacts": [review], "artifact_status": "ready",
              "artifact_bundle": {"artifacts": [review], "status": "ready"},
              "outcome_axes": {"artifacts": {"status": "ready"}}}
    write_task_result(canonical, "applied", "completed", child_drive_root=str(child), **fields)
    write_task_result(child, "applied", "completed", **fields)
    for materialize in (False, True):
        effective = load_effective_task_result(canonical, "applied", materialize_artifacts=materialize)
        assert effective["artifacts"] == []
        assert effective["artifact_bundle"]["status"] == "not_applicable"
    copied = copy_child_task_result(canonical, {"id": "applied", "drive_root": str(child)})
    assert copied["artifacts"] == []
    assert copied["artifact_status"] == "not_applicable"
    assert copied["artifact_bundle"]["status"] == "not_applicable"
    assert (artifacts.task_artifact_dir_path(canonical, "applied") / source["path"]).read_bytes() == b"canonical source"


def test_unregistered_review_names_and_completion_custody_are_not_filtered(tmp_path):
    store = artifacts.task_artifact_dir_path(tmp_path, "applied", create=True)
    (store / "acceptance-user.json").write_text("user file", encoding="utf-8")
    (store / "notes.lock").write_text("user lock", encoding="utf-8")
    artifacts.store_task_artifact_bytes(tmp_path, "applied", "completion.json", b"retained555", kind="task_completion_observations")
    artifacts.store_task_artifact_bytes(tmp_path, "applied", "acceptance-host.json", b"host551", kind="task_acceptance_review")
    nested = store / "user-folder"
    nested.mkdir()
    (nested / "acceptance-host.json").write_text("unregistered same basename", encoding="utf-8")
    records = artifacts.collect_task_artifact_records(tmp_path, "applied")
    assert {row["path"] for row in records} == {str(store / "acceptance-user.json"), str(store / "notes.lock"),
                                               str(store / "completion.json"), str(nested / "acceptance-host.json")}


def test_materialized_user_artifact_outweighs_stale_review_only_readiness(tmp_path):
    source = artifacts.store_task_artifact_bytes(tmp_path, "applied", "review.json", b"source", kind="task_acceptance_review")
    review = artifacts.artifact_record(artifacts.task_artifact_dir_path(tmp_path, "applied") / source["path"], kind="task_acceptance_review")
    write_task_result(tmp_path, "applied", "completed", artifacts=[review], artifact_status="ready",
                      artifact_bundle={"artifacts": [review], "status": "ready"},
                      outcome_axes={"artifacts": {"status": "ready"}})
    artifacts.store_task_artifact_bytes(tmp_path, "applied", "answer.txt", b"real result", kind="user_file")

    effective = load_effective_task_result(tmp_path, "applied")

    assert [item["name"] for item in effective["artifacts"]] == ["answer.txt"]
    assert effective["artifact_bundle"]["status"] == "ready"
    assert effective["artifact_status"] == "ready"
    assert effective["outcome_axes"]["artifacts"]["status"] == "ready"


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
