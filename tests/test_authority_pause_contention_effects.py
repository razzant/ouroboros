"""Save contention after paid/tool effects must not replay either effect."""
import json

import pytest

from ouroboros import budget_pause as bp, usage_accounting as ua
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.task_results import write_task_result
from ouroboros.task_store import patch_document
from tests._usage_store_testing import ledger_rows
from tests.test_authority_pause_contention import _contended_save
from tests.test_billing_authority_unavailable import _loop_driver, _run

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("stage", ["seed", "checkpoint"])
def test_paid_response_and_completed_tool_survive_save_contention_once(tmp_path, monkeypatch, stage):
    write_task_result(tmp_path, "member", "running", root_task_id="member")
    loop, registry, _ = _loop_driver(tmp_path, monkeypatch, "member")
    sends, effects, calls = [], [], []

    def tool(*_args, **_kwargs):
        effects.append("effect-once")
        # A corrupt whole-work binding: the next admission rereads it as an unreadable authority.
        patch_document(tmp_path, "member", lambda row: {**row, "billing_group": {"billing_group_id": "member"}})
        return "completed-effect-once"

    registry.override_handler("knowledge_read", tool)

    def model(call):
        calls.append(True)
        response = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "paid-read", "type": "function", "function": {
                "name": "knowledge_read", "arguments": '{"topic":"fixture"}'}}]}
        ua.execute_physical_attempt(
            ua.AttemptRequest(model="fixture", provider="fixture", reservation_usd=0.1),
            lambda: sends.append(response) or response,
            extractor=lambda _response: ({"prompt_tokens": 1, "completion_tokens": 1}, 0.01, True))
        return response, 0.01, call.active_context_mode

    monkeypatch.setattr(loop, "_call_round_model", model)
    attempts, failures, holds, release = _contended_save(tmp_path, monkeypatch, stage, "begin")
    try:
        with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="member", root_task_id="member",
                                        root_limit_usd=20, global_limit_usd=1000)):
            with pytest.raises(bp.BudgetPauseRequested) as parked:
                _run(loop, registry, tmp_path, "member")
        assert len(sends) == 1 and effects == ["effect-once"]
        assert len(calls) == 2  # the second admission refuses before a physical send
        assert len(failures) == len(holds) == 1 and len(attempts) == 2
        assert attempts[0] is attempts[1]
        records = ledger_rows(tmp_path)
        assert len(records) == 1 and records[0]["state"] == "settled"
        state = json.loads(read_actor_source_bytes(tmp_path, "member", parked.value.pause["source_ref"]))
        results = [m for m in state["messages"] if m.get("tool_call_id") == "paid-read"]
        assert len(results) == 1 and "completed-effect-once" in str(results[0]["content"])
        assert records[0]["cost_usd"] == pytest.approx(0.01)
        assert parked.value.pause["resume_point"]["unanswered_tool_call_ids"] == []
    finally:
        release()
        bp.end_dispatch_fence("member")
