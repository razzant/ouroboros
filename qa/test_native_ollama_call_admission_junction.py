"""Junction controls: native Ollama (/api/chat) replies through Development's call admission (#4272).

Measurement-only module, copied into a snapshot's tests/ directory; never part of the candidate.
Fictional data except the recorded 0.40.16 canary stream the candidate already carries.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

TARGET = {"provider": "openai-compatible", "resolved_model": "qwen-stand:latest"}
CANARY = Path(__file__).parent / "fixtures/llm_wire/ollama/ollama-0.40.16-synthetic-canary.ndjson"


def _frame(content="", *, calls=None, done=False, reason="stop", thinking=""):
    message = {"role": "assistant", "content": content}
    if thinking:
        message["thinking"] = thinking
    if calls is not None:
        message["tool_calls"] = calls
    frame = {"model": "qwen-stand:latest", "message": message, "done": done}
    if done:
        frame.update(done_reason=reason, prompt_eval_count=120, eval_count=7)
    return frame


def _native(frames):
    """A native reply as chat_ollama returns it: Reply assembly, then the adapter's own usage projection."""
    from ouroboros.llm_ollama import _result
    from ouroboros.llm_ollama_wire import Reply

    reply = Reply()
    for frame in frames:
        if isinstance(frame, (bytes, bytearray)):
            reply.add_line(frame)
        else:
            reply.add(frame)
    return _result(reply.response(), dict(TARGET))


def _admit(msg, usage, logs):
    """The merged round seam: response facts, the round's projection, then call admission."""
    from ouroboros.loop_llm_call import _replace_response_meta, _take_response_facts
    from ouroboros.loop_output_exhaustion import admit_calls, response_facts

    ctx = SimpleNamespace(task_id="junction", model_turn_state=None, _current_llm_call_meta={})
    accumulated: dict = {}
    later = _take_response_facts(usage, msg, accumulated)
    ctx._request_wire_custom_receipts = accumulated.pop("_request_wire_custom_receipts", ())
    _replace_response_meta(response_facts(ctx), usage, msg, later_output_limit=later)
    admitted, calls = admit_calls(ctx, msg, logs)
    return ctx, accumulated, later, admitted, calls


# --- real /api/chat data -------------------------------------------------------------------------------------

def test_recorded_canary_text_reply_passes_the_merged_seam_unchanged(tmp_path):
    lines = [line for line in CANARY.read_bytes().split(b"\n") if line.strip()]
    msg, usage = _native(lines)
    ctx, accumulated, later, admitted, calls = _admit(msg, usage, tmp_path)
    assert admitted is msg and calls == [] and later == ""
    assert admitted["content"] == "ЛАЗУРНЫЙ_КАПИБАРА_917" and "tool_calls" not in admitted
    assert ctx._round_response_facts == {"finish_reason_present": True, "finish_reason": "stop", "tool_call_count": 0}
    assert (usage["prompt_tokens"], usage["completion_tokens"]) == (44_498, 16)
    assert not (tmp_path / "events.jsonl").exists()


# --- structured native arguments -----------------------------------------------------------------------------

def test_native_structured_arguments_are_admitted_byte_for_byte(tmp_path):
    from ouroboros.loop_output_exhaustion import unfinished_finish

    arguments = {"path": "notes.txt", "text": "line1\nline2\tcol\r", "n": 3}
    msg, usage = _native([_frame(thinking="plan"), _frame(calls=[{"function": {"name": "write_file", "arguments": arguments}}]),
                          _frame(done=True)])
    raw_arguments = msg["tool_calls"][0]["function"]["arguments"]
    ctx, _acc, later, admitted, calls = _admit(msg, usage, tmp_path)
    assert usage["response_finish_reason"] == "tool_calls" and later == ""
    assert admitted is msg and calls == msg["tool_calls"]
    assert calls[0]["function"]["arguments"] == raw_arguments and json.loads(raw_arguments) == arguments
    assert ctx._round_response_facts["tool_call_count"] == 1 and unfinished_finish(ctx, admitted) == ""
    assert not (tmp_path / "events.jsonl").exists()  # nothing to repair: native arguments are already strict JSON


def test_native_batch_cut_at_its_output_limit_starts_no_call(tmp_path):
    from ouroboros.loop_output_exhaustion import _cut_call_finish

    msg, usage = _native([_frame(calls=[{"function": {"name": "read_file", "arguments": {"path": "a"}}}]),
                          _frame(done=True, reason="length")])
    ctx, _acc, _later, admitted, calls = _admit(msg, usage, tmp_path)
    assert usage["response_finish_reason"] == "length"
    assert calls == [] and admitted is msg and _cut_call_finish(ctx, admitted) == "length"


def test_a_cut_native_batch_transcript_replays_on_native_ollama(tmp_path):
    """#4272 keeps the cut batch's whole native calls answered NOT STARTED; the next native request carries them."""
    from ouroboros.llm_ollama_wire import native_messages
    from ouroboros.loop_output_exhaustion import intermediate_reply

    msg, usage = _native([_frame("Reading both.", calls=[
        {"function": {"name": "read_file", "arguments": {"path": "a.txt"}}},
        {"function": {"name": "read_file", "arguments": {"path": "b.txt"}}}]), _frame(done=True, reason="length")])
    tool_ctx, _acc, _later, admitted, calls = _admit(msg, usage, tmp_path)
    assert calls == []
    tool_ctx.task_id = ""  # no durable task root: the gating save writes nothing and waits for nothing
    said = []
    ctx = SimpleNamespace(tools=SimpleNamespace(_ctx=tool_ctx), messages=[{"role": "user", "content": "read a and b"}],
                          accumulated_usage={}, llm_trace={"reasoning_notes": []}, budget_tail="tool")
    assert intermediate_reply(ctx, admitted, lambda text, **_kw: said.append(text)) is True
    assert ctx.budget_tail == "no_tool" and said == ["Reading both."]
    assistant, *results = ctx.messages[1:]
    assert [call["function"]["arguments"] for call in assistant["tool_calls"]] == ['{"path": "a.txt"}', '{"path": "b.txt"}']
    assert all(row["role"] == "tool" and row["content"].startswith("[HOST NOTICE] NOT STARTED") for row in results)
    rows = native_messages(ctx.messages)
    assert [call["function"]["arguments"] for call in rows[1]["tool_calls"]] == [{"path": "a.txt"}, {"path": "b.txt"}]
    assert [row["tool_name"] for row in rows[2:]] == ["read_file", "read_file"]
    assert all(set(row) <= {"role", "content", "tool_name", "tool_call_id"} for row in rows[2:])


@pytest.mark.parametrize("bad", [
    {"calls": {"function": {"name": "x", "arguments": {}}}},          # tool_calls not a list
    {"calls": [{"function": {"name": "x", "arguments": "{\"a\": 1}"}}]},  # arguments not an object
    {"calls": [{"function": {"name": "", "arguments": {}}}]},          # a call without a name
])
def test_native_malformed_calls_are_refused_before_admission(bad):
    from ouroboros.llm_ollama_wire import OllamaIntegrityError

    with pytest.raises(OllamaIntegrityError):
        _native([_frame(calls=bad["calls"]), _frame(done=True)])


# --- the conflicted seam itself -------------------------------------------------------------------------------

@pytest.mark.parametrize("calls,expected", [
    ([{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}], "same"),
    ({"id": "c1"}, []),
    ("call", []),
    (None, []),
], ids=["list", "dict", "string", "absent"])
def test_receipts_bind_only_a_list_and_route_facts_are_always_taken(monkeypatch, calls, expected):
    from ouroboros import loop_llm_call

    seen = []
    monkeypatch.setattr(loop_llm_call, "pop_custom_validation_receipts", lambda usage, tool_calls: seen.append(tool_calls) or ())
    msg = {"role": "assistant", "content": "ok", **({"tool_calls": calls} if calls is not None else {})}
    usage = {"route_notice": "notice A", "route_caution": "caution A", "route_caution_key": "route-a",
             "_chat_later_output_limit": "length"}
    accumulated: dict = {}
    assert loop_llm_call._take_response_facts(usage, msg, accumulated) == "length"
    assert seen == [calls if expected == "same" else expected] and (expected != "same" or seen[0] is calls)
    assert accumulated["_route_notice"] == "notice A" and accumulated["_last_route_notice"] == "notice A"
    assert [entry["words"] for entry in accumulated["_route_cautions_pending"]] == ["caution A"]
    assert "_request_wire_custom_receipts" not in accumulated and "_chat_later_output_limit" not in usage


def test_a_reply_without_calls_keeps_the_ordinary_text_path(tmp_path):
    from ouroboros.loop_output_exhaustion import unfinished_finish

    msg, usage = _native([_frame("Hello"), _frame(" there", done=True)])
    ctx, accumulated, later, admitted, calls = _admit(msg, usage, tmp_path)
    assert (admitted, calls, later) == (msg, [], "") and admitted["content"] == "Hello there"
    assert unfinished_finish(ctx, admitted) == "" and "_route_cautions_pending" not in accumulated
    cut, cut_usage = _native([_frame("Hel"), _frame(done=True, reason="length")])
    cut_ctx, _a, _l, _m, cut_calls = _admit(cut, cut_usage, tmp_path)
    assert cut_calls == [] and unfinished_finish(cut_ctx, cut) == "length"  # the existing unfinished-reply episode


# --- history written by #4272, read by the native projection ---------------------------------------------------

def _compatible_reply(arguments: str):
    msg = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "write_file", "arguments": arguments}}]}
    return msg, {"response_finish_reason": "tool_calls"}


def test_repaired_history_replays_on_native_ollama(tmp_path):
    from ouroboros.llm_ollama_wire import native_messages

    msg, usage = _compatible_reply('{"path": "n.txt", "text": "a\nb"}')  # literal LF inside a string value
    _ctx, _acc, _later, admitted, calls = _admit(msg, usage, tmp_path)
    assert admitted is not msg and calls[0]["function"]["arguments"] == '{"path": "n.txt", "text": "a\\nb"}'
    assert [json.loads(line)["type"] for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()] == [
        "tool_arguments_repaired"]
    rows = native_messages([{"role": "user", "content": "write"}, admitted,
                            {"role": "tool", "tool_call_id": "call_1", "content": "ok"}])
    assert rows[1]["tool_calls"][0]["function"]["arguments"] == {"path": "n.txt", "text": "a\nb"}
    assert rows[2]["tool_name"] == "write_file"


@pytest.mark.parametrize("arguments", ['{"path": "n.txt", "text": "a\nb"}', '{"path": "n.t'], ids=["raw-control", "cut"])
def test_native_projection_refuses_history_arguments_that_are_not_json(arguments):
    """Unchanged boundary of the native route: a row it cannot carry losslessly fails before dispatch."""
    from ouroboros.llm_ollama_wire import OllamaUnsupportedRequest, native_messages

    msg, _usage = _compatible_reply(arguments)
    with pytest.raises(OllamaUnsupportedRequest, match="valid JSON"):
        native_messages([{"role": "user", "content": "write"}, msg])
