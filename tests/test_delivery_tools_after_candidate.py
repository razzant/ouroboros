"""A retained answer constrains final text without disabling further tool work."""

import json

from tests.test_delivery_candidate import _run_loop


def test_further_write_then_complete_replacement_remain_available(tmp_path, monkeypatch):
    replacement = "The requested effect.txt now contains the verified result."
    result, _usage, trace, calls = _run_loop(
        tmp_path, monkeypatch,
        ["The first answer is retained.",
         {"content": None, "tool_calls": [{"id": "write-1", "type": "function", "function": {
             "name": "write_file", "arguments": json.dumps({"path": "effect.txt", "content": "verified result"}),
         }}]},
         json.dumps({"delivery_control": "replace", "full_answer": replacement})],
        acceptance_results=[True, False],
    )
    prompt = json.dumps(calls[1])
    assert "may continue using tools" in prompt
    assert "only to your final response with no tool calls" in prompt
    assert (tmp_path / "effect.txt").read_text() == "verified result"
    assert result == replacement
    assert trace["delivery_candidate"]["revision"] == 2
    assert trace["delivery_candidate"]["finalization_control"] == "replace"
    assert trace["delivery_candidate"]["degraded"] is False
