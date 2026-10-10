"""The ordinary publication formatter yields to Pause before publication starts."""
from __future__ import annotations

import json
import threading

import pytest

from ouroboros import budget_pause, loop, model_wait, usage_accounting as ua
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.observability import call_manifest_path
from ouroboros.task_results import write_task_result
from ouroboros.tools import skill_publish
from tests._budget_pause_exact_helpers import _install_queue
from tests._usage_store_testing import attempt_rows_in_start_order
from tests.test_loop_transport_wait import _loop_kwargs
from tests.test_pause_author_aux_receivers import _start_loop
from tests.test_pause_model_receiver import _request, _extract, _until
from tests.test_skill_publish_transaction import _install_transaction_fakes, _snapshot
from tests.test_working_checkpoint import _registry

pytestmark = pytest.mark.serial


def test_pause_of_real_tool_formatter_checkpoints_blocked_attempt_before_late_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GITHUB_TOKEN", "inert-test-token")
    monkeypatch.setattr("ouroboros._usage_wait.ABANDON_POLL_SEC", 0.02)
    budget_pause.end_dispatch_fence("t-wait")
    _ctx, events, _captured = _install_transaction_fakes(monkeypatch, tmp_path, snapshot=_snapshot())
    # Real registry target resolution, with every remote/scanner boundary owned
    # by the existing inert publication fixture. No provider or GitHub is called.
    skill = tmp_path / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: demo\ndescription: inert fixture\nversion: 1.0.0\ntype: instruction\n---\n# Demo\n")
    registry = _registry(tmp_path, monkeypatch, "t-wait", 1)
    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "t-wait", "running", root_task_id="t-wait")
    workers.RUNNING["t-wait"] = {"task": {"id": "t-wait", "type": "task", "chat_id": 0,
                                          "root_task_id": "t-wait"}, "worker_id": 0, "attempt": 1}
    sent, release = threading.Event(), threading.Event()
    calls = []

    def sender():
        sent.set()
        assert release.wait(10)
        return {"content": "late optional PR formatter body"}

    class Formatter:
        def chat(self, **kwargs):
            assert model_wait.current_model_wait() is registry._ctx.model_wait_context
            calls.append(kwargs)
            return ua.execute_physical_attempt(_request(), sender, extractor=_extract), {}

    monkeypatch.setattr(skill_publish, "LLMClient", Formatter)
    main_calls = []

    def model(*args, **kwargs):
        main_calls.append(True)
        assert len(main_calls) == 1, "the author bought another model call before pausing"
        return {"role": "assistant", "content": "", "tool_calls": [{"id": "formatter", "type": "function",
            "function": {"name": "submit_skill_to_hub", "arguments": json.dumps({
                "skill": "demo", "confirm_public_submission": True})}}]}, 0.0

    monkeypatch.setattr(loop, "_dispatch_round_model", model)
    thread, returned, outcome = _start_loop(tmp_path, _loop_kwargs(tmp_path, registry, []))
    try:
        assert sent.wait(5), repr(outcome)
        from supervisor.owner_pause_control import request_owner_pause

        ack = request_owner_pause("t-wait", request_id="pause-formatter")
        assert ack["ok"], ack
        assert returned.wait(3), "the ordinary formatter kept the author Pausing until its sender answered"
        assert isinstance(outcome.get("error"), budget_pause.BudgetPauseRequested), repr(outcome)
        assert "result" not in outcome and len(calls) == len(main_calls) == 1
        assert not any(event[0] == "mutation" for event in events)
        row = attempt_rows_in_start_order(tmp_path)[0]
        assert row["state"] == "dispatched"
        saved = json.loads(read_actor_source_bytes(tmp_path, "t-wait", outcome["error"].pause["source_ref"]))
        tool = next(row for row in saved["trace"]["tool_calls"] if row.get("tool_call_id") == "formatter")
        # The normal transcript may carry a bounded view. Its exact producer
        # source remains the parser's authority, including the new attempt IDs.
        result = json.loads(read_actor_source_bytes(tmp_path, "t-wait", tool["producer_source_ref"]))
        assert result["status"] == "blocked" and result["reason_code"] == "owner_pause"
        assert result["completed_stage"] == "upstream_read"
        assert [effect["stage"] for effect in result["completed_effects"]] == [
            "local_validation", "snapshot_captured", "local_preflight", "upstream_read"]
        assert result["completed_effects"][-1]["base_sha"] == "1" * 40
        assert result["abandoned_model_attempt_ids"].split(",") == [row["attempt_id"]]
        assert result["model_outcome"] == "unknown" and not result.get("receipt")
        assert saved["delivery_candidate"] is None, "a blocked tool result is not a task final"
        assert "late optional PR formatter body" not in str(saved)
        stopped_events = list(events)
    finally:
        release.set()
        thread.join(5)
        budget_pause.end_dispatch_fence("t-wait")
    assert not thread.is_alive()
    assert _until(lambda: attempt_rows_in_start_order(tmp_path)[0]["state"] == "settled")
    path = call_manifest_path(tmp_path, "t-wait", f"physical_{row['attempt_id']}_response")
    assert _until(path.exists), "the late sender keeps the paid response as evidence"
    assert events == stopped_events, "late formatter text triggered another scan or publication step"
