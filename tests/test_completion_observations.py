"""Synthesis sees task observations without mistaking tool success for receipt."""

import hashlib
import gzip
import json
import shutil
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import agent_task_pipeline as pipeline
from ouroboros.artifacts import task_artifact_dir_path, store_task_artifact_bytes
from ouroboros.gateway.tasks import api_task_artifact
from ouroboros.headless import copy_child_task_result
from ouroboros.projects_registry import bind_task_to_project, create_project
from ouroboros.task_finalization import build_completion_observations, build_sealed_final_package, sealed_final_prompt_section
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.task_status import load_effective_task_result


def _trace():
    return {"tool_calls": [{"tool": "send_photo", "tool_call_id": "photo1", "status": "ok",
                            "result": "OK: photo sent to owner chat."}] + [
        {"tool": "send_user_message", "tool_call_id": f"message{i}", "status": "ok",
         "result": f"queued message {i}"} for i in range(40)], "reasoning_notes": []}


@pytest.mark.parametrize("mode", ["root", "project", "child", "project_child"])
@pytest.mark.parametrize("user_file", [False, True])
def test_sealed_observations_survive_copyback_and_recovery_without_global_attribution(tmp_path, monkeypatch, mode, user_file):
    source_root = tmp_path / "execution"
    canonical = tmp_path / "canonical" if "child" in mode else source_root
    source_root.mkdir()
    canonical.mkdir(exist_ok=True)
    if "project" in mode:
        create_project(canonical, "viewer-project")
        bind_task_to_project(canonical, "observation", "viewer-project", origin={"absent": "system"})
    if source_root != canonical:
        write_task_result(canonical, "observation", "running", child_drive_root=str(source_root))
    if user_file:
        store_task_artifact_bytes(source_root, "observation", "answer.txt", b"genuine result", kind="user_file")
    seen = []
    def related_skills(root, trace, root_task_id, **kwargs):
        seen.append((root, root_task_id))
        kwargs["history_coverage"].update(complete=True)
        return [{"name": "viewer", "enabled": True, "ready": True, "blockers": []}]
    monkeypatch.setattr("ouroboros.skill_readiness.acceptance_skill_lifecycle", related_skills)
    env = SimpleNamespace(drive_root=source_root, repo_dir=source_root)
    task = {"id": "observation", "root_task_id": "observation", "type": "task",
            "text": "Show the completed viewer."}
    if source_root != canonical:
        task["budget_drive_root"] = str(canonical)
    pipeline._store_task_result(env, task, "", {"rounds": 4, "cost": 0}, _trace())
    stored = load_task_result(source_root, "observation")
    observations = stored["completion_observations"]
    assert seen == [(str(canonical) if source_root != canonical else canonical, "observation")]
    assert observations["delivery_counts"]["send_photo"] == {"calls": 1, "reported_ok": 1, "status_unknown": 0}
    assert observations["delivery_counts"]["send_user_message"]["calls"] == 40
    assert observations["delivery_results_omitted"] == 39
    assert any(row["tool"] == "send_photo" for row in observations["delivery_results"])
    assert observations["delivery_receipt_coverage"] == "not_observed_by_tool_trace"
    assert "not action attribution" in observations["skill_state_scope"]
    ref = observations["source_ref"]
    raw = (task_artifact_dir_path(canonical, "observation") / ref["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ref["sha256"]
    assert len(json.loads(raw)["delivery_results"]) == 41
    expected = "ready" if user_file else "not_applicable"
    assert stored["artifact_bundle"]["status"] == expected
    assert [row["name"] for row in stored["artifacts"]] == (["answer.txt"] if user_file else [])
    def download():
        app = Starlette(routes=[Route("/api/tasks/{task_id}/artifacts/{name}", api_task_artifact)])
        app.state.drive_root = canonical
        with TestClient(app) as client:
            response = client.get(f"/api/tasks/observation/artifacts/{ref['path']}")
            assert response.status_code == 200 and response.content == raw
    download()  # canonical custody exists before any child artifact copy-back
    if source_root != canonical:
        assert not (task_artifact_dir_path(source_root, "observation") / ref["path"]).exists()
        copied = copy_child_task_result(canonical, {"id": "observation", "drive_root": str(source_root)})
        shutil.rmtree(source_root)
    else:
        copied = stored
    assert (task_artifact_dir_path(canonical, "observation") / ref["path"]).read_bytes() == raw
    for materialize in (False, True):
        effective = load_effective_task_result(canonical, "observation", materialize_artifacts=materialize)
        assert effective["artifact_bundle"]["status"] == expected
        assert effective["outcome_axes"]["artifacts"]["status"] == expected
    download()
    if user_file:
        assert (task_artifact_dir_path(canonical, "observation") / "answer.txt").read_bytes() == b"genuine result"
    # Recovery uses the stored observations, without replaying an absent trace.
    sealed = build_sealed_final_package(copied, "")
    assert sealed["completion_observations"] == observations
    assert not any(row["name"] == ref["path"] for row in sealed["artifact_manifest"])
    prompt = sealed_final_prompt_section(sealed)
    assert "send_photo" in prompt and "photo sent to owner chat" in prompt
    assert "not a chat receipt" in prompt
    assert "not evidence that no action occurred" in prompt
    assert "viewer" in prompt and '"ready": true' in prompt
    assert "does not attribute an owner's click" in prompt


@pytest.mark.parametrize("failure", [OSError, ValueError, TimeoutError])
def test_full_source_failure_is_disclosed_without_inventing_a_ref(tmp_path, monkeypatch, failure):
    monkeypatch.setattr("ouroboros.skill_readiness.acceptance_skill_lifecycle", lambda *_a, **_k: [])
    def fail(*args, **kwargs):
        raise failure("source unavailable")
    monkeypatch.setattr("ouroboros.artifacts.store_task_artifact_bytes", fail)
    trace = {"tool_calls": [{"tool": "send_photo", "status": "error", "is_error": True,
                             "result_partial": True, "result": "submission incomplete"}]}
    result = build_completion_observations(tmp_path, {"id": "partial"}, trace)
    assert result["source_status"] == "unavailable" and "source_ref" not in result
    assert result["delivery_results"][0]["result_partial"] is True
    assert result["delivery_results"][0]["is_error"] is True
    assert result["delivery_counts"]["send_photo"]["reported_ok"] == 0


def test_empty_and_legacy_observations_remain_unknown_without_an_extra_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.skill_readiness.acceptance_skill_lifecycle", lambda *_a, **_k: [])
    empty = build_completion_observations(tmp_path, {"id": "empty"}, {"tool_calls": []})
    assert empty["trace_available"] is True and empty["delivery_counts"] == {}
    assert "source_ref" not in empty
    assert not task_artifact_dir_path(tmp_path, "empty").exists()
    legacy = sealed_final_prompt_section(build_sealed_final_package({}, ""))
    assert '"status": "unavailable"' in legacy
    assert "not evidence that no action occurred" in legacy


def test_packet_summary_receives_inline_facts_and_custodies_its_prompt(tmp_path, monkeypatch):
    from ouroboros.observability import OBSERVABILITY_DIR
    import ouroboros.post_task_synthesis as synthesis

    monkeypatch.setattr("ouroboros.skill_readiness.acceptance_skill_lifecycle", lambda *_a, **_k: [])
    monkeypatch.setattr("ouroboros.consolidator._consolidation_route", lambda: ("test-model", False))
    # The existing trace builder has already applied its 4000-character bound;
    # a second 3000-character head cap used to discard this later fact again.
    monkeypatch.setattr(synthesis, "build_trace_summary", lambda trace: "x" * 3200 + " LATE_TRACE_FACT")
    task = {"id": "summary", "root_task_id": "summary", "chat_id": 1, "type": "task", "text": "show it"}
    observations = build_completion_observations(tmp_path, task, _trace())
    sealed = build_sealed_final_package({"completion_observations": observations}, "")
    prompts = []
    class CapturingLLM:
        def chat(self, **kwargs):
            prompts.append(kwargs["messages"][0]["content"])
            return {"content": "Photo submission was recorded; owner receipt is unknown."}, {"cost": 0}
    pipeline._run_task_summary(SimpleNamespace(drive_root=tmp_path), CapturingLLM(), task,
                               {"rounds": 4, "cost": 0}, _trace(), tmp_path / "logs", sealed_final=sealed)
    assert len(prompts) == 1 and "LATE_TRACE_FACT" in prompts[0]
    assert "send_photo" in prompts[0] and "not a chat receipt" in prompts[0]
    calls_root = tmp_path / OBSERVABILITY_DIR / "calls"
    manifests = [json.loads(path.read_text()) for path in calls_root.glob("*/*.json")]
    observed = [row for row in manifests if row.get("call_type") == "task_summary"]
    request = next(row for row in observed if row["call_id"].endswith("_request"))
    response = next(row for row in observed if row["call_id"].endswith("_response"))
    with gzip.open(request["full_payload_ref"]["path"], "rt") as source:
        assert json.load(source)["kwargs"]["messages"][0]["content"] == prompts[0]
    with gzip.open(response["full_payload_ref"]["path"], "rt") as source:
        assert "owner receipt is unknown" in json.load(source)["message"]["content"]
