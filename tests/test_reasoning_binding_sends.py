"""Conversation-bound reasoning is retired at the real canonical send seams."""

from __future__ import annotations

import base64
import copy
import json
import queue
from types import SimpleNamespace

import pytest

from ouroboros import loop, loop_forced_finalization as forced, task_pacing
from ouroboros.transcript_prefix import observe_send, retire_unbound_reasoning
from tests._dialogue import chat_rows, progress_rows
from tests.test_consolidator_context_fit import fit as _consolidation_fit
from tests.test_native_review_context_facts import _executor
from tests.test_native_tool_round_executor import _ScriptedLLM, _tool_call, _VERDICT
from tests.test_reasoning_binding import TOOLS, bound_rows, history, signed
from tests.test_transcript_prefix import ANSWER, _four_tool_rounds, full_loop as _full_loop
from tests.test_tree_cost_ceiling import _ctx

consolidation_fit = _consolidation_fit
full_loop = _full_loop


def _reasoned_rounds(fixture):
    inner = _four_tool_rounds(fixture)

    def main(*args, **kwargs):
        message, cost = inner(*args, **kwargs)
        text = f"reasoning-{fixture.model_step}"
        return {**message, "reasoning": text, "reasoning_details": [signed(text)]}, cost

    return main


def _retirement_events(fixture):
    return [row["data"] for row in list(fixture.events.queue)
            if isinstance(row, dict) and isinstance(row.get("data"), dict)
            and row["data"].get("checkpoint_kind") == "prompt_prefix_break"
            and row["data"].get("reasoning_retired")]


def test_real_append_only_loop_keeps_every_answered_reasoning_turn(full_loop, monkeypatch):
    monkeypatch.setattr(loop, "call_llm_with_retry", _reasoned_rounds(full_loop))
    result, usage, _trace = full_loop.run()
    assert result == ANSWER, (result, full_loop.progress)
    counts = [len(bound_rows(messages)) for messages in full_loop.model_inputs]
    assert counts == sorted(counts) and counts[-1] >= 3
    assert _retirement_events(full_loop) == []
    assert usage.get("reasoning_retired", 0) == 0


def test_real_compaction_retires_the_invalidated_tail_before_the_next_send(full_loop, monkeypatch):
    from ouroboros.tools.compact_context import _compact_context

    reasoned = _reasoned_rounds(full_loop)

    def main(*args, **kwargs):
        response = reasoned(*args, **kwargs)
        if full_loop.model_step == 2:
            _compact_context(full_loop.ctx, working_note="The first reads are folded.", keep_unit_ids=[])
        return response

    monkeypatch.setattr(loop, "call_llm_with_retry", main)
    result, usage, _trace = full_loop.run()
    assert result == ANSWER, (result, full_loop.progress)
    before, folded, next_round = full_loop.model_inputs[1:4]
    assert len(bound_rows(before)) == 1
    assert bound_rows(folded) == []
    assert len(bound_rows(next_round)) == 1
    events = _retirement_events(full_loop)
    assert [(event["round"], event["reasoning_retired"]["cause"]) for event in events] == [(3, "messages")]
    assert usage["reasoning_retired"] == 1


def _prepared_context(tmp_path, *, changed):
    messages = history()
    owner = SimpleNamespace(task_id="task1", drive_root=tmp_path, task_metadata={})
    observe_send(owner, messages[:6], tools=TOOLS, round_idx=1)
    if changed:
        messages[3]["content"] = "The earlier tool result was rewritten."
    return _ctx(messages=messages, tools=SimpleNamespace(_ctx=owner), tool_schemas=copy.deepcopy(TOOLS),
                drive_root=tmp_path, drive_logs=tmp_path / "logs", llm=SimpleNamespace())


class _ReachedMeasurement(BaseException):
    """Stop at the actual forced-call measurement boundary, outside failure handling."""


@pytest.mark.parametrize("changed", [False, True])
def test_forced_final_retires_before_appending_its_prompt_and_measuring(tmp_path, monkeypatch, changed):
    ctx = _prepared_context(tmp_path, changed=changed)
    seen = {}
    prompt = "Give the final answer now."
    real_append = loop._append_or_merge_user_message

    def append(messages, text, *args, **kwargs):
        if text == prompt:
            seen["before_prompt"] = copy.deepcopy(messages)
        return real_append(messages, text, *args, **kwargs)

    def measure(context, messages):
        assert context is ctx
        seen["measurement"] = copy.deepcopy(messages)
        raise _ReachedMeasurement()

    monkeypatch.setattr(loop, "_append_or_merge_user_message", append)
    monkeypatch.setattr(forced, "_forced_physical_context", measure)
    with pytest.raises(_ReachedMeasurement):
        forced._forced_final_answer(ctx, prompt=prompt, fallback_text="fallback", reason_code="round_limit",
                                    _prompt_prepared=True)
    expected = [2] if changed else [2, 4, 6]
    assert bound_rows(seen["before_prompt"]) == bound_rows(seen["measurement"]) == expected
    assert prompt not in json.dumps(seen["before_prompt"])
    assert prompt in json.dumps(seen["measurement"])
    assert ctx.accumulated_usage.get("reasoning_retired", 0) == int(changed)


class _ReachedSecondForcedSend(BaseException):
    """Stop after the real forced owner-refresh path has prepared its second send."""


@pytest.mark.parametrize("images,incoming_image,remaining,retired", [
    (10, False, 10, False), (9, True, 10, False), (10, True, 5, True),
])
def test_forced_owner_refresh_rechecks_reasoning_after_real_image_delivery(
    tmp_path, monkeypatch, images, incoming_image, remaining, retired,
):
    from tests.test_live_image_delivery import pixels

    ctx = _prepared_context(tmp_path, changed=False)
    encoded = base64.b64encode(pixels()).decode()
    image = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
    ctx.messages.insert(2, {"role": "user", "content": [copy.deepcopy(image) for _ in range(images)]})
    observe_send(ctx.tools._ctx, ctx.messages[:-2], round_idx=1, tools=TOOLS)
    ctx.incoming_messages = queue.Queue()
    sent, measured = [], []

    def measure(_ctx, messages):
        measured.append(copy.deepcopy(messages))

    def chat(_llm, messages, *_args, **_kwargs):
        sent.append(copy.deepcopy(messages))
        if len(sent) == 2:
            raise _ReachedSecondForcedSend()
        incoming = {"text": "Use the new owner input in the final answer."}
        if incoming_image:
            incoming.update(image_base64=encoded, image_mime="image/png")
        ctx.incoming_messages.put(incoming)
        return {"role": "assistant", "content": "First draft."}, 0.0

    monkeypatch.setattr(forced, "_forced_physical_context", measure)
    monkeypatch.setattr(loop, "_delivery_evidence_state", lambda *_a, **_kw: None)
    monkeypatch.setattr(loop, "_finalize_forced_services", lambda *_a, **_kw: None)
    monkeypatch.setattr(forced, "_forced_subject_prompt", lambda *_a, **_kw: "")
    monkeypatch.setattr(loop, "call_llm_with_retry", chat)
    with pytest.raises(_ReachedSecondForcedSend):
        forced._forced_final_answer(ctx, prompt="Give the final answer.", fallback_text="fallback",
                                    reason_code="round_limit", _prompt_prepared=True)
    assert bound_rows(sent[0]) == [3, 5, 7]
    live_images = [block for row in sent[1] if isinstance(row.get("content"), list)
                   for block in row["content"] if isinstance(block, dict)
                   and block.get("type") in {"image", "image_url"}]
    assert len(live_images) == remaining
    assert bound_rows(sent[1]) == bound_rows(measured[1]) == ([] if retired else [3, 5, 7])
    assert "[FORCED_OWNER_REFRESH]" in json.dumps(sent[1])
    assert ctx.accumulated_usage.get("reasoning_retired", 0) == int(retired)
    facts = [row["content"] for row in sent[1] if isinstance(row.get("content"), str)
             and "Earlier reasoning not replayed:" in row["content"]]
    assert len(facts) == int(retired)
    if retired:
        assert "[CONTEXT_FACTS]" in facts[0]
        assert "your signed reasoning from 3 later turns" in facts[0]
        assert "your visible text, tool calls and results are unchanged" in facts[0]


@pytest.mark.parametrize("changed", [False, True])
def test_wrapup_retires_before_send_projection_measurement_and_pricing(tmp_path, monkeypatch, changed):
    from ouroboros import loop_llm_call

    ctx = _prepared_context(tmp_path, changed=changed)
    seen, priced_request = {}, object()

    def prepare(messages, **_kwargs):
        seen["projection"] = copy.deepcopy(messages)
        return copy.deepcopy(messages)

    def measure(_ctx, messages):
        seen["measurement"] = copy.deepcopy(messages)
        return None

    def price(**kwargs):
        seen["price"] = copy.deepcopy(kwargs["messages"])
        return priced_request

    monkeypatch.setattr(loop_llm_call, "_prepare_main_messages", prepare)
    monkeypatch.setattr(forced, "_forced_physical_context", measure)
    monkeypatch.setattr(task_pacing, "prospective_wrapup_attempt_request", price)
    prompt = "Give the final answer now."
    prospective = [dict(message) for message in ctx.messages]
    loop._append_or_merge_user_message(prospective, prompt)
    request, prepared = task_pacing.prepared_wrapup_candidate(ctx, prospective, allow_server_web_search=False)
    assert request is priced_request
    expected = [2] if changed else [2, 4, 6]
    assert all(bound_rows(messages) == expected for messages in [*seen.values(), prepared, ctx.messages, prospective])
    assert list(seen) == ["projection", "measurement", "price"]
    assert ctx.accumulated_usage.get("reasoning_retired", 0) == int(changed)

    def forced_measurement(_ctx, messages):
        seen["fresh_forced"] = copy.deepcopy(messages)
        raise _ReachedMeasurement()

    monkeypatch.setattr(forced, "_forced_physical_context", forced_measurement)
    with pytest.raises(_ReachedMeasurement):
        forced._forced_final_answer(ctx, prompt=prompt, fallback_text="fallback", reason_code="budget_exhausted",
                                    _prompt_prepared=True)
    assert seen["fresh_forced"] == seen["price"]
    assert ctx.accumulated_usage.get("reasoning_retired", 0) == int(changed)


@pytest.mark.parametrize("changed", [False, True])
def test_waiting_main_retires_before_refreshing_facts_and_measuring(tmp_path, monkeypatch, changed):
    from ouroboros import loop_messages, loop_model_call

    ctx = _prepared_context(tmp_path, changed=changed)
    ctx.context_fit_plan, ctx.active_context_mode = None, "max"
    seen = {}
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda *_args, **_kwargs: (None, "max"))
    monkeypatch.setattr(loop_model_call, "_fit_route_tool_ceiling", lambda _ctx: False)
    monkeypatch.setattr(loop_model_call, "_pending_model_wait_handover", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loop_model_call, "_project_wake_input", lambda _ctx: None)
    monkeypatch.setattr(loop_messages, "append_context_facts",
                        lambda context: seen.update(facts=copy.deepcopy(context.messages)))

    def measure(context, **_kwargs):
        seen["measurement"] = copy.deepcopy(context.messages)
        raise _ReachedMeasurement()

    monkeypatch.setattr(loop, "_measure_round_main_fit", measure)
    with pytest.raises(_ReachedMeasurement):
        loop_model_call._reprepare_waiting_main(ctx, {"model": ctx.active_model, "model_role": "main"})
    expected = [2] if changed else [2, 4, 6]
    assert list(seen) == ["facts", "measurement"]
    assert all(bound_rows(messages) == expected for messages in seen.values())
    assert ctx.accumulated_usage.get("reasoning_retired", 0) == int(changed)


def _signed_call(name, arguments, call_id):
    return {"reasoning": call_id, "reasoning_details": [signed(call_id)],
            "tool_calls": [_tool_call(name, arguments, call_id)]}


def _helper_retirement_logs(context, count):
    from ouroboros.tools.registry import ToolContext

    assert isinstance(context, ToolContext) and callable(context.drive_logs)
    path = context.drive_logs() / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    retired = [row for row in rows if row.get("reasoning_retired")]
    assert len(retired) == count
    for row in retired:
        assert row["type"] == "task_checkpoint" and row["checkpoint_kind"] == "prompt_prefix_break"
        assert row["task_id"] == context.task_id
        assert not {"system_type", "text", "card_row", "card_row_id", "progress_row_emitted"} & row.keys()
    assert progress_rows(context.drive_root) == []
    assert chat_rows(context.drive_root) == []
    return retired


@pytest.mark.parametrize("surface", ["multi_model_review", "deep_self_review"])
@pytest.mark.parametrize("fold", [False, True])
def test_native_review_retires_before_chat_and_records_only_logs(tmp_path, fold, surface):
    second = (_signed_call("compact_context", {"working_note": "The source remains in its checkpoint.", "keep_unit_ids": []}, "fold")
              if fold else _signed_call("read_file", {"path": "source.txt"}, "read-again"))
    llm = _ScriptedLLM([
        _signed_call("read_file", {"path": "source.txt"}, "read"), second, {"content": _VERDICT},
    ])
    executor = _executor(tmp_path, llm)
    executor.assignment.request.surface = surface
    result = executor.execute()
    assert result.raw_text == _VERDICT and len(llm.calls) == 3
    assert len(bound_rows(llm.calls[1]["messages"])) == 1
    assert len(bound_rows(llm.calls[2]["messages"])) == (0 if fold else 2)
    assert result.usage.get("reasoning_retired", 0) == int(fold)
    _helper_retirement_logs(executor._inspection_ctx, int(fold))
    if fold:
        assert executor._view_receipt["status"] == "applied"
        assert "Earlier reasoning not replayed:" in json.dumps(llm.calls[2]["messages"])


@pytest.mark.parametrize("changed", [False, True])
def test_failed_native_send_records_the_checked_view_for_metadata_and_later_binding(tmp_path, changed):
    from ouroboros.context_compaction import context_reclaim_transcript_sha256
    from ouroboros.review_native_episode import _wire_size

    error = RuntimeError("The provider did not answer.")
    error.physical_attempt_capture = SimpleNamespace(state="settled")

    class RefusingLLM(_ScriptedLLM):
        def chat(self, **kwargs):
            if len(self.calls) == 2:
                self.calls.append({**kwargs, "messages": copy.deepcopy(kwargs["messages"])})
                raise error
            return super().chat(**kwargs)

    llm = RefusingLLM([
        _signed_call("read_file", {"path": "source.txt"}, "read-1"),
        _signed_call("read_file", {"path": "source.txt"}, "read-2"),
    ])
    executor = _executor(tmp_path, llm)
    with pytest.raises(RuntimeError) as raised:
        executor.execute()
    assert raised.value is error
    assert len(llm.calls) == 3
    latest = llm.calls[-1]
    observed = executor._inspection_ctx._last_context_observation
    assert observed["messages"] == latest["messages"] != llm.calls[-2]["messages"]
    assert observed["tool_schemas"] == latest["tools"]
    usage = executor._episode_usage
    assert usage["native_rounds"] == 3
    assert usage["native_context"]["view_revision"] == context_reclaim_transcript_sha256(latest["messages"])
    assert usage["native_context"]["serialized_chars"] == usage["native_transcript_chars"] == _wire_size(
        latest["messages"], latest["tools"])
    assert usage["native_context"]["previous_request"]["round"] == 2
    replay = copy.deepcopy(latest["messages"])
    assert bound_rows(replay) == [2, 4]
    if changed:
        replay[3]["content"] = "The earlier read result changed after dispatch."
    fact = retire_unbound_reasoning(executor._inspection_ctx, replay, latest["tools"])
    assert fact == ({"index": 3, "cause": "messages", "turns": 1, "blocks": 1} if changed else None)
    assert bound_rows(replay) == ([2] if changed else [2, 4])


@pytest.mark.parametrize("changed", [False, True])
def test_waiting_native_review_retires_against_its_sent_observation(tmp_path, monkeypatch, changed):
    from ouroboros.tools.compact_context import record_context_view
    from ouroboros.tools.registry import ToolContext

    executor = _executor(tmp_path, _ScriptedLLM([]), sources=False)
    messages = history()
    executor._inspection_ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path / "data", task_id="native-reprepare")
    record_context_view(executor._inspection_ctx, messages[:6], TOOLS)
    if changed:
        messages[3]["content"] = "The earlier tool result was rewritten."
    monkeypatch.setattr(executor, "_refresh_window", lambda _route: None)
    prepared = executor._reprepare_model_call({
        "model": executor.assignment.slot.model, "messages": copy.deepcopy(messages), "tools": copy.deepcopy(TOOLS),
    }, messages=messages)
    assert prepared["messages"] is messages
    assert bound_rows(messages) == ([2] if changed else [2, 4, 6])
    assert executor._episode_usage.get("reasoning_retired", 0) == int(changed)
    _helper_retirement_logs(executor._inspection_ctx, int(changed))


def _knowledge(tmp_path):
    from ouroboros.consolidator import KnowledgeReadContext
    from ouroboros.tools.registry import ToolContext

    repo = tmp_path / "subject"
    repo.mkdir()
    (repo / "source.txt").write_text("The complete source remains recoverable.\n", encoding="utf-8")
    return KnowledgeReadContext(ToolContext(repo_dir=repo, drive_root=tmp_path / "data", task_id="light-binding"))


@pytest.mark.parametrize("fold", [False, True])
def test_consolidator_retires_after_authored_compaction_before_observed_chat(tmp_path, consolidation_fit, fold):
    from ouroboros import consolidator

    knowledge = _knowledge(tmp_path)
    second = (_signed_call("compact_context", {"working_note": "The source remains in its checkpoint.", "keep_unit_ids": []}, "fold")
              if fold else _signed_call("read_file", {"path": "source.txt"}, "read-again"))
    llm = _ScriptedLLM([
        _signed_call("read_file", {"path": "source.txt"}, "read"), second, {"content": "Summary of the source."},
    ])
    content, usage = consolidator._call_consolidation_llm(llm, "Summarize the source.", "Probe", knowledge=knowledge)
    assert content == "Summary of the source.", usage
    assert len(llm.calls) == 3
    assert len(bound_rows(llm.calls[1]["messages"])) == 1
    assert len(bound_rows(llm.calls[2]["messages"])) == (0 if fold else 2)
    assert usage.get("reasoning_retired", 0) == int(fold)
    _helper_retirement_logs(knowledge.context, int(fold))
    if fold:
        assert "Earlier reasoning not replayed:" in json.dumps(llm.calls[2]["messages"])
        result = next(row["content"] for row in llm.calls[2]["messages"] if row.get("tool_call_id") == "fold")
        assert json.loads(result)["context_view"]["status"] == "applied"


def test_consolidator_accumulates_usage_and_separate_log_facts_across_answered_rounds(tmp_path, consolidation_fit):
    from ouroboros import consolidator

    knowledge = _knowledge(tmp_path)
    llm = _ScriptedLLM([
        _signed_call("read_file", {"path": "source.txt"}, "read-1"),
        _signed_call("compact_context", {"working_note": "The first source remains recoverable.", "keep_unit_ids": []}, "fold-1"),
        _signed_call("read_file", {"path": "source.txt"}, "read-2"),
        _signed_call("compact_context", {"working_note": "Both reads agree and remain recoverable.", "keep_unit_ids": []}, "fold-2"),
        {"content": "Summary of the source."},
    ])
    content, usage = consolidator._call_consolidation_llm(llm, "Summarize the source.", "Probe", knowledge=knowledge)
    assert content == "Summary of the source." and len(llm.calls) == 5
    assert usage["reasoning_retired"] == usage["reasoning_retired_turns"] == usage["reasoning_retired_blocks"] == 2
    events = _helper_retirement_logs(knowledge.context, 2)
    assert [event["round"] for event in events] == [3, 5]
    assert all(event["reasoning_retired"]["turns"] == event["reasoning_retired"]["blocks"] == 1 for event in events)


def test_unanswered_consolidator_send_does_not_replace_the_answered_baseline(tmp_path, consolidation_fit):
    from ouroboros import consolidator

    class RefusingLLM(_ScriptedLLM):
        def chat(self, **kwargs):
            if self.calls:
                self.calls.append({**kwargs, "messages": copy.deepcopy(kwargs["messages"])})
                raise RuntimeError("The provider did not answer.")
            return super().chat(**kwargs)

    knowledge = _knowledge(tmp_path)
    llm = RefusingLLM([_signed_call("read_file", {"path": "source.txt"}, "read")])
    content, usage = consolidator._call_consolidation_llm(llm, "Summarize the source.", "Probe", knowledge=knowledge)
    assert not content and usage["_consolidation_errors"]
    assert len(llm.calls) == 2
    assert knowledge.context._last_context_observation["messages"] == llm.calls[0]["messages"]
    assert knowledge.context._last_context_observation["messages"] != llm.calls[1]["messages"]
