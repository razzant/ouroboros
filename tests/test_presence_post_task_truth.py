"""Memory synthesis remembers reply preparation separately from diagnostics."""

from types import SimpleNamespace

import pytest

from ouroboros import agent_task_pipeline as pipeline
from ouroboros.task_finalization import build_sealed_final_package, sealed_final_prompt_section
from ouroboros.task_results import load_task_result, write_task_result


@pytest.mark.parametrize("completion, origin, failed, work_ref, expected", [
    (None, "model_final", False, "", "message"),
    (None, "host_notice", True, "", "silent"),
    (None, "host_salvage", True, "child-work", "deferred"),
    ("silent", "model_final", False, "", "silent"),
    ("tool_delivered", "model_final", False, "", "tool_delivered"),
    ("deferred", "model_final", False, "child-work", "deferred"),
])
def test_live_and_recovered_synthesis_use_recorded_presence_body(
    tmp_path, monkeypatch, completion, origin, failed, work_ref, expected,
):
    captured = []
    launch = pipeline._run_post_task_processing_async

    sealed_inputs = []

    def run_synchronously(*args, **kwargs):
        kwargs["blocking"] = True
        sealed_inputs.append(kwargs["sealed_final"])
        return launch(*args, **kwargs)

    def paid_stage(env, task, *_args, **_kwargs):
        # Exercise the real claim and its durable running phase before the
        # fake paid stage runs on the phase that received the sealed input.
        current = load_task_result(env.drive_root, task["id"])
        assert current["root_phase_checkpoint"]["post_task_synthesis"] == "running"
        captured.append(sealed_inputs[-1])

    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", run_synchronously)
    monkeypatch.setattr("ouroboros.llm.LLMClient", lambda: SimpleNamespace())
    monkeypatch.setattr(pipeline, "_run_memory_fallback_draft", paid_stage)
    raw = "Internal terminal diagnostic" if failed else "Current authored result"
    task = {"id": "presence-synthesis", "root_task_id": "presence-synthesis", "type": "presence",
            "chat_id": 7, "text": "Read the request", "_is_direct_chat": True,
            "metadata": {"presence": {"binding_id": "binding"}}}
    ctx = SimpleNamespace(_presence_completion={"outcome": completion} if completion else None,
                          _presence_completion_accepted=bool(completion),
                          _swarm_handoff_attempt={"status": "scheduled", "task_id": work_ref} if work_ref else None)
    usage = {"terminal_origin": origin, "terminal_provider_notice": "Internal provider status"}
    if failed:
        usage.update(execution_status="failed", reason_code="round_limit")
    events = []
    pipeline.emit_task_results(
        SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path), None, None, events, task, raw,
        usage, {"tool_calls": [], "reasoning_notes": []}, 0.0, tmp_path / "logs", ctx=ctx,
    )
    event = next(row for row in events if row["type"] == "presence_result")
    stored = load_task_result(tmp_path, task["id"])
    assert event["outcome"] == expected
    assert stored["result"] == raw
    assert len(captured) == 1
    sealed = captured[0]
    assert sealed["final_result_text"] == event["text"]
    assert sealed["internal_terminal_text"] == raw
    assert sealed["presence_delivery"] == {
        "reply_recorded": True, "outcome": expected, "work_ref": work_ref,
        "terminal_origin": origin, "status": stored["status"],
        "reason_code": stored.get("reason_code") or "unknown",
    }
    prompt = sealed_final_prompt_section(sealed)
    assert "not a provider delivery receipt" in prompt
    assert "Final result text (submitted for delivery)" not in prompt
    assert "Internal provider status" in prompt
    if not event["text"]:
        assert "Recorded automatic-reply body:\n(empty final message)\nInternal terminal text" in prompt
        assert raw in prompt.split("Internal terminal text (separate from the automatic reply):\n", 1)[1]
    else:
        assert f"Recorded automatic-reply body:\n{raw}\n" in prompt

    # The ordinary restart path reconstructs the same event-time evidence;
    # it must not use the raw diagnostic as a fallback for an empty body.
    write_task_result(tmp_path, task["id"], stored["status"],
                      root_phase_checkpoint={"post_task_synthesis": "pending_once"})
    assert pipeline.recover_pending_root_post_task_synthesis(tmp_path, tmp_path) == 1
    assert len(captured) == 2 and captured[1] == sealed
    assert load_task_result(tmp_path, task["id"])["root_phase_checkpoint"]["post_task_synthesis"] == "completed"


def test_historical_host_reply_is_remembered_without_claiming_delivery():
    row = {"task_id": "old-turn", "status": "failed", "reason_code": "round_limit",
           "terminal_origin": "host_notice", "result": "Historic host diagnostic",
           "metadata": {"presence": {}, "presence_outcome": "message",
                        "presence_result_text": "Historic host diagnostic"}}
    sealed = build_sealed_final_package(row, row["result"])
    assert sealed["final_result_text"] == "Historic host diagnostic"
    assert sealed["presence_delivery"]["terminal_origin"] == "host_notice"
    assert sealed["presence_delivery"]["outcome"] == "message"
    assert "not a provider delivery receipt" in sealed_final_prompt_section(sealed)
    # Today's retry policy prevents a new send; it cannot erase biography.
    from ouroboros.presence_runner import presence_result_from_stored

    assert presence_result_from_stored(row, "old-turn").text == ""


def test_missing_automatic_reply_record_is_unknown_not_raw_fallback():
    row = {"status": "failed", "result": "Internal-only text", "metadata": {
        "presence": {}, "presence_outcome": "deferred", "presence_work_ref": "child"}}
    sealed = build_sealed_final_package(row, row["result"])
    assert sealed["final_result_text"] == ""
    assert sealed["presence_delivery"]["reply_recorded"] is False
    assert sealed["presence_delivery"]["outcome"] == "deferred"
    assert sealed["presence_delivery"]["work_ref"] == "child"
    assert sealed["presence_delivery"]["terminal_origin"] == "unknown"
    prompt = sealed_final_prompt_section(sealed)
    assert "(automatic reply record unavailable)" in prompt and "Internal-only text" in prompt


def test_ordinary_owner_synthesis_retains_its_existing_package_and_prompt():
    sealed = build_sealed_final_package({"terminal_origin": "model_final"}, "Owner answer")
    assert sealed == {"final_result_text": "Owner answer", "artifact_manifest": [],
                      "completion_observations": {"status": "unavailable"}}
    prompt = sealed_final_prompt_section(sealed)
    assert "Final result text (submitted for delivery):\nOwner answer\n" in prompt
    assert "Presence" not in prompt and "Internal terminal text" not in prompt
