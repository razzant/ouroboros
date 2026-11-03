"""A saved Git candidate is not necessarily dirty; completion never guesses from custody."""
from types import SimpleNamespace

import pytest

from ouroboros import mutation_attribution as attribution
from ouroboros.outcomes import derive_loop_outcome
from ouroboros.project_dialogue import _completion_verdict, outcome_phase
from ouroboros.task_results import STATUS_RUNNING, write_task_result
from ouroboros.terminal_projection import _project_row
from tests.test_mutation_attribution import _git, _repo


def _start(tmp_path):
    repo, data = _repo(tmp_path), tmp_path / "data"
    (repo / "dirty.txt").write_text("foreign change\n")
    write_task_result(data, "candidate-task", STATUS_RUNNING)
    attribution.capture_mutation_baseline(data, "candidate-task", [{"surface_type": "system_repo", "host_root": str(repo)}])
    (repo / "clean.txt").write_text("task commit\n")
    _git(repo, "add", "clean.txt")
    _git(repo, "commit", "-qm", "task")
    attribution.advance_mutation_baseline(data, "candidate-task", repo)
    return repo, data


def _fact(data):
    evidence = attribution.record_terminal_mutation_candidates(data, "candidate-task")
    return attribution.mutation_evidence_projection(evidence)["uncommitted_remainder"]


def test_commit_then_edit_projects_only_loose_owned_paths_through_terminal(tmp_path):
    repo, data = _start(tmp_path)
    (repo / "clean.txt").write_text("after commit\n")
    (repo / "new.py").write_text("value = 1\n")
    (repo / ".gitignore").write_text(".review-evidence/\n")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-qm", "ignore evidence")
    attribution.advance_mutation_baseline(data, "candidate-task", repo)
    (repo / ".review-evidence").mkdir()
    (repo / ".review-evidence" / "log.txt").write_text("not product\n")
    fact = _fact(data)
    from ouroboros.task_results import load_task_result
    snapshot = load_task_result(data, "candidate-task")["mutation_evidence"]["terminal_candidate_snapshot"]
    assert fact == {"state": "known", "count": 2, "paths_omitted": 2,
        "snapshot_at": snapshot["captured_at"], "baseline_hash": snapshot["baseline_hash"]}
    assert snapshot["surfaces"][0]["canonical_root"] == str(repo)
    assert snapshot["surfaces"][0]["base_commit"] == _git(repo, "rev-parse", "HEAD")
    assert snapshot["surfaces"][0]["uncommitted_paths"] == ["clean.txt", "new.py"]
    projection = attribution.load_mutation_evidence_projection(data, "candidate-task")
    outcome = derive_loop_outcome("Saved the candidate.", {"tool_calls": 1}, {"mutation_attribution": projection})
    record = {"status": "completed", "outcome_axes": outcome["outcome_axes"], "reason_code": "final_message"}
    assert outcome_phase(record, {}) == "done"
    assert "2 uncommitted files remain in the saved candidate" in _completion_verdict(record, {})
    row = _project_row("candidate-task", record, {}, {"chat_id": 42})
    assert row["outcome"] == "Done"
    assert row["outcome_axes"]["execution"]["mutation_attribution"]["uncommitted_remainder"] == {**fact, "task_id": "candidate-task"}
    assert "2 uncommitted files" in row["text"]


def test_clean_saved_commit_is_not_reported_as_uncommitted(tmp_path):
    _repo_path, data = _start(tmp_path)
    fact = _fact(data)
    assert fact["state"] == "known" and fact["count"] == 0
    record = {"status": "completed", "outcome_axes": {"execution": {"mutation_attribution": {
        "terminal_candidate_count": 9, "uncommitted_remainder": fact}}}}
    assert "uncommitted" not in _completion_verdict(record, {}).lower()


def test_unreadable_git_is_unknown_not_zero(tmp_path, monkeypatch):
    _repo_path, data = _start(tmp_path)
    monkeypatch.setattr(attribution, "_git_status_paths", lambda _: (_ for _ in ()).throw(OSError("unreadable")))
    fact = _fact(data)
    assert fact["state"] == "unknown" and fact["count"] is None
    record = {"status": "completed", "outcome_axes": {"execution": {"mutation_attribution": {"uncommitted_remainder": fact}}}}
    assert outcome_phase(record, {}) == "done"
    assert "could not be checked" in _completion_verdict(record, {})


def test_failed_durable_capture_cannot_reuse_a_previous_clean_snapshot(tmp_path, monkeypatch):
    from ouroboros.agent_task_pipeline import _attach_host_mutation_projection
    _repo_path, data = _start(tmp_path)
    _fact(data)
    monkeypatch.setattr(attribution, "record_terminal_mutation_candidates",
                        lambda *_: (_ for _ in ()).throw(OSError("disk unavailable")))
    trace = {}
    _attach_host_mutation_projection(SimpleNamespace(drive_root=data), {"id": "candidate-task"}, trace)
    assert trace["mutation_attribution"]["uncommitted_remainder"]["state"] == "unknown"


def test_legacy_snapshot_does_not_invent_dirty_files(tmp_path):
    _repo_path, data = _start(tmp_path)
    evidence = attribution.record_terminal_mutation_candidates(data, "candidate-task")
    for surface in evidence["terminal_candidate_snapshot"]["surfaces"]:
        surface.pop("uncommitted_paths")
        surface.pop("uncommitted_state")
    assert "uncommitted_remainder" not in attribution.mutation_evidence_projection(evidence)


@pytest.mark.parametrize("malformed", [[], "unknown", {"surfaces": None}])
def test_unreadable_old_snapshot_does_not_break_the_projection(malformed):
    projection = attribution.mutation_evidence_projection({"terminal_candidate_snapshot": malformed})
    assert not projection["clean_eligible"]


def test_large_remainder_is_compact_but_exact_paths_are_readable_from_task_details(tmp_path):
    import json
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient
    from ouroboros.gateway.tasks import api_task_get
    from ouroboros.task_results import load_task_result

    repo, data = _start(tmp_path)
    evidence = attribution.record_terminal_mutation_candidates(data, "candidate-task")
    # A large recorded observation tests the wire without creating 5,000 scratch files.
    paths = [f"generated/{i:05d}/long file `literal` <name>.py" for i in range(5000)]
    surface = evidence["terminal_candidate_snapshot"]["surfaces"][0]
    surface["uncommitted_paths"] = paths
    surface.update(uncommitted_state="known", blockers=[])
    write_task_result(data, "candidate-task", "completed", mutation_evidence=evidence)
    projection = attribution.load_mutation_evidence_projection(data, "candidate-task")
    assert len(json.dumps(projection).encode()) < 2048
    fact = projection["uncommitted_remainder"]
    assert fact["count"] == fact["paths_omitted"] == 5000
    assert fact["task_id"] == "candidate-task"
    outcome = derive_loop_outcome("Saved.", {}, {"mutation_attribution": projection})
    written = write_task_result(data, "candidate-task", "completed", outcome_axes=outcome["outcome_axes"])
    terminal = _project_row("candidate-task", written, {}, {"chat_id": 1})
    assert len(json.dumps(terminal).encode()) < 4096
    assert terminal["outcome"] == "Done" and "5000 uncommitted files" in terminal["text"]
    assert load_task_result(data, "candidate-task")["mutation_evidence"] == evidence
    app = Starlette(routes=[Route("/api/tasks/{task_id}", api_task_get)])
    app.state.drive_root = data
    with TestClient(app) as client:
        response = client.get("/api/tasks/" + fact["task_id"])
    assert response.status_code == 200
    detail = response.json()["mutation_evidence"]["terminal_candidate_snapshot"]
    assert detail["captured_at"] == fact["snapshot_at"]
    assert detail["baseline_hash"] == fact["baseline_hash"]
    assert detail["surfaces"][0]["canonical_root"] == str(repo)
    assert detail["surfaces"][0]["uncommitted_paths"] == paths
