"""Ordinary synthetic tool results and answered transcript witnesses."""
from copy import deepcopy

from ouroboros import context_compaction as cc
from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY, make_tool_result_record

BODY = "Exact producer bytes: Ж, e\u0301, emoji 🧠.\r\n" * 1000 + "CONSEQUENTIAL-TAIL\r\n"


def _tool(call_id="reused", *, invocation=None, facts=None, text=BODY):
    result = {"role": "tool", "tool_call_id": call_id, "content": text}
    if invocation is not None:
        result[TOOL_RESULT_RECORD_KEY] = make_tool_result_record(
            {"invocation_id": invocation, "tool_call_id": call_id, "execution_id": "exec",
             "round_id": "round", "task_attempt": 1,
             "trace_ref": {"manifest_ref": {"path": f"trace/{invocation}.json", "sha256": "a" * 64}}},
            text, facts=facts, source_ref={"path": f"source/{invocation}.txt"})
    return [{"role": "assistant", "content": "", "tool_calls": [{
        "id": call_id, "type": "function", "function": {"name": "run_command", "arguments": '{"cmd":["fixture"]}'}}]},
        result]


def _observation(messages):
    return {"messages": deepcopy(messages), "exposed_units": [
        {"unit_id": u.unit_id, "raw_sha256": u.raw_sha256} for u in cc.context_units(messages, scope="dialogue")]}
