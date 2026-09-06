"""Synthesis sees task observations without mistaking tool success for receipt."""

import hashlib
import gzip
import json
import shutil
from types import SimpleNamespace

from ouroboros import agent_task_pipeline as pipeline
from ouroboros.artifacts import task_artifact_dir_path
from ouroboros.headless import copy_child_task_result
from ouroboros.task_finalization import build_completion_observations, build_sealed_final_package, sealed_final_prompt_section
from ouroboros.task_results import load_task_result


def _trace():
    return {"tool_calls": [{"tool": "send_photo", "tool_call_id": "photo1", "status": "ok",
                            "result": "OK: photo sent to owner chat."}] + [
        {"tool": "send_user_message", "tool_call_id": f"message{i}", "status": "ok",
         "result": f"queued message {i}"} for i in range(40)], "reasoning_notes": []}


def test_sealed_observations_survive_copyback_and_recovery_without_global_attribution(tmp_path, monkeypatch):
    source_root, canonical = tmp_path / "execution", tmp_path / "canonical"
    source_root.mkdir()
    seen = []
    def related_skills(root, trace, root_task_id, **kwargs):
        seen.append((root, root_task_id))
        kwargs["history_coverage"].update(complete=True)
        return [{"name": "viewer", "enabled": True, "ready": True, "blockers": []}]
    monkeypatch.setattr("ouroboros.skill_readiness.acceptance_skill_lifecycle", related_skills)
    env = SimpleNamespace(drive_root=source_root, repo_dir=source_root)
    task = {"id": "observation", "root_task_id": "observation", "type": "task",
            "budget_drive_root": str(canonical), "text": "Show the completed viewer."}
    pipeline._store_task_result(env, task, "", {"rounds": 4, "cost": 0}, _trace())
    stored = load_task_result(source_root, "observation")
    observations = stored["completion_observations"]
    assert seen == [(str(canonical), "observation")]
    assert observations["delivery_counts"]["send_photo"] == {"calls": 1, "reported_ok": 1, "status_unknown": 0}
    assert observations["delivery_counts"]["send_user_message"]["calls"] == 40
    assert observations["delivery_results_omitted"] == 39
    assert any(row["tool"] == "send_photo" for row in observations["delivery_results"])
    assert observations["delivery_receipt_coverage"] == "not_observed_by_tool_trace"
    assert "not action attribution" in observations["skill_state_scope"]
    ref = observations["source_ref"]
    raw = (task_artifact_dir_path(source_root, "observation") / ref["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ref["sha256"]
    assert len(json.loads(raw)["delivery_results"]) == 41
    copied = copy_child_task_result(canonical, {"id": "observation", "drive_root": str(source_root)})
    shutil.rmtree(source_root)
    assert (task_artifact_dir_path(canonical, "observation") / ref["path"]).read_bytes() == raw
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
