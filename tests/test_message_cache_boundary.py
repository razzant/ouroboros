"""The one message-side cache boundary across real rounds and every lane (#596, #1186).

``context_fit.seal_task_transcript`` marks the newest user or tool row with text, in
the loop's order: seal, then this round's facts row, then the consumed clock line and
the answer. Only a marker-carrying send copy renders the mark; every other lane sends
canonical bytes. Each check reads the payload the lane itself builds."""

from __future__ import annotations

import copy
import json

import pytest

from ouroboros import context_compaction as cc
from ouroboros.context_budget import (
    HOST_CONTEXT_KIND_KEY,
    HOST_TEXT_SHA_KEY,
    OWNER_WORDS_KIND,
    bound_host_row,
    host_row_intact,
)
from ouroboros.context_fit import seal_task_transcript
from ouroboros.context_source_view import _obsolete_host_rows
from ouroboros.llm import LLMClient
from ouroboros.llm_claudexor import _request as claudexor_request
from ouroboros.llm_messages import STABLE_PREFIX_BLOCKS_KEY
from ouroboros.loop_messages import CONTEXT_FACTS_NAME
from ouroboros.send_clock import CLOCK_ROW_NAME
from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY, make_tool_result_record, read_tool_result_record
from tests.test_review_prompt_caching import _openrouter_target, _send_direct_anthropic

_TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "read",
                                            "parameters": {"type": "object", "properties": {}}}}]
_IMAGE = [{"type": "text", "text": "screenshot"},
          {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]
# Per round: owner words drained before it, then the answer's tool outputs (None: a text answer).
_ROUNDS = (
    (None, ["one result"]),
    (None, ["batch a", "", "batch c"]),  # a batch with an empty output inside it
    (None, None),                        # tool-free: the clock line and an answer (#1186)
    ("And the docs?", None),             # owner words, then another answer
    (None, [""]),                        # an empty output is never the anchor
    (None, [_IMAGE]),                    # nor an opaque one
    (None, ["after the gap"]),
    (None, None),
)
# The newest user/tool row with text when each round seals: the task, a tool result with no
# lag, the end of the batch, owner words, a clock line past an answer, past empty/opaque results.
_EXPECTED = (1, 5, 11, 15, 17, 20, 24, 30)
_MARKED = ("anthropic", "openrouter-anthropic", "openrouter-openai")
_PLAIN = ("openrouter-grok", "openai", "claudexor", "local", "gigachat")
_OPENROUTER = {"openrouter-anthropic": "anthropic/claude-fable-5", "openrouter-openai": "openai/gpt-6.1-sol",
               "openrouter-grok": "x-ai/grok-4.7"}


def _system(book: bool) -> dict:
    stable = ["governance " * 900, *(["## DEVELOPMENT.md\nrules"] if book else []), "identity and story"]
    return {"role": "system", STABLE_PREFIX_BLOCKS_KEY: len(stable), "content": [
        *({"type": "text", "text": text, "cache_control": {"type": "ephemeral"}} for text in stable),
        {"type": "text", "text": "runtime facts"}]}


def _rounds(book: bool):
    """Yield ``(round, sealed transcript as sent, the same rows never sealed)`` per round."""
    messages = [_system(book), {"role": "user", "content": "task: compare the reports"}]
    shadow = copy.deepcopy(messages)

    def add(row):
        messages.append(row)
        shadow.append(copy.deepcopy(row))

    for number, (owner, outputs) in enumerate(_ROUNDS, start=1):
        if owner:
            add({"role": "user", HOST_CONTEXT_KIND_KEY: OWNER_WORDS_KIND, "content": owner})
        seal_task_transcript(messages)
        add(bound_host_row(CONTEXT_FACTS_NAME, f"[facts] round {number}"))
        yield number, messages, shadow
        add(bound_host_row(CLOCK_ROW_NAME, f"[Host clock at this request: 2026-10-21T00:0{number}:00Z UTC]"))
        if outputs is None:
            add({"role": "assistant", "content": f"answer {number}"})
            continue
        calls = [f"call-{number}-{index}" for index in range(len(outputs))]
        add({"role": "assistant", "content": "", "tool_calls": [
            {"id": call, "type": "function", "function": {"name": "read_file", "arguments": "{}"}} for call in calls]})
        for call, output in zip(calls, outputs):
            row = {"role": "tool", "tool_call_id": call, "content": copy.deepcopy(output)}
            if isinstance(output, str):
                row[TOOL_RESULT_RECORD_KEY] = make_tool_result_record(
                    {"tool_call_id": call, "invocation_id": f"inv-{call}"}, output, facts={"status": "ok"})
            add(row)


def _unmarked(messages: list) -> list:
    return [{key: value for key, value in m.items() if key != "cache_control"} for m in messages]


def _payload(lane: str, messages: list, monkeypatch) -> dict:
    client = LLMClient(api_key="unused")
    if lane == "anthropic":
        payload, usage = _send_direct_anthropic(monkeypatch, messages, _TOOLS)  # the JSON requests posts
        assert usage.get("prompt_cache_breakpoints_reduced") is None
        return payload
    if lane == "claudexor":
        return claudexor_request({"source": "codex", "resolved_model": "model"}, messages, None,
                                 {"model_account_override": ""})
    if lane == "local":
        return client._build_local_candidate(messages, _TOOLS, 512, "auto")[1]
    if lane == "gigachat":
        return {"messages": LLMClient._gigachat_messages(messages)}
    target = (_openrouter_target(_OPENROUTER[lane]) if lane in _OPENROUTER else
              {"provider": "openai", "resolved_model": "gpt-5.5", "usage_model": "openai/gpt-5.5",
               "supports_openrouter_extensions": False})
    payload = client._build_remote_kwargs(target, messages, "high", 512, "auto", None, _TOOLS,
                                          skip_capability_fetch=True)
    client._normalize_payload_cache_ttl(target, payload)
    assert client._cache_breakpoint_tls.pending is None
    return payload


def _marked_texts(payload: dict) -> list:
    """Every message-side block marker's text, nested tool results included."""
    found = []
    for message in payload.get("messages") or []:
        assert "cache_control" not in message, "the host-only row mark never reaches a wire"
        for block in message["content"] if message.get("role") != "system" and isinstance(
                message.get("content"), list) else ():
            nested = block.get("content") if isinstance(block, dict) and block.get("type") == "tool_result" else None
            for holder in (block, *(nested if isinstance(nested, list) else ())):
                if isinstance(holder, dict) and "cache_control" in holder:
                    found.append(holder.get("text"))
    return found


def _blocks(payload: dict) -> tuple:
    """``(wire order without markers, positions of message-side markers)``. A string and its one
    text block count as one form: the moving tool-result seal has always relied on that."""
    out, marks = [], []
    for message in payload["messages"]:
        content = message.get("content")
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])
        for part in [*parts, *({"type": "tool_call", **call} for call in message.get("tool_calls") or [])]:
            nested = part.get("content") if part.get("type") == "tool_result" else None
            nested = nested if isinstance(nested, list) else []
            if message["role"] != "system" and any("cache_control" in holder for holder in (part, *nested)):
                marks.append(len(out))
            part = {key: value for key, value in part.items() if key != "cache_control"}
            if nested and all(block.get("type") == "text" for block in nested):
                part["content"] = "".join(block["text"] for block in nested)
            out.append(json.dumps([message["role"], message.get("tool_call_id"), part], sort_keys=True))
    return out, marks


def _wire(payload: dict) -> list:
    return [json.dumps(message, sort_keys=True, ensure_ascii=False) for message in payload["messages"]]


def _expected_text(row: dict) -> str:
    content = row["content"]
    return content if isinstance(content, str) else [b["text"] for b in content if b.get("type") == "text"][-1]


@pytest.mark.parametrize("book", [False, True], ids=["no-handbook", "handbook"])
def test_each_round_seals_the_newest_text_row_and_only_marker_lanes_render_it(monkeypatch, book):
    seen, observed = {lane: [] for lane in (*_MARKED, *_PLAIN)}, []
    for number, messages, shadow in _rounds(book):
        expected = _EXPECTED[number - 1]
        assert [i for i, m in enumerate(messages) if "cache_control" in m] == [expected], number
        assert _unmarked(messages) == shadow, "the seal never changes canonical content"
        # A view authored over the last observed send still binds: the move is presentation only.
        assert cc._view_source_messages(messages[:len(observed)]) == cc._view_source_messages(observed)
        observed = copy.deepcopy(messages)
        hosts = [i for i, m in enumerate(messages) if HOST_TEXT_SHA_KEY in m]
        assert all(host_row_intact(messages[i]) for i in hosts)
        newest = {max(i for i in hosts if messages[i][HOST_CONTEXT_KIND_KEY] == kind)
                  for kind in {messages[i][HOST_CONTEXT_KIND_KEY] for i in hosts}}
        assert _obsolete_host_rows(messages) == set(hosts) - newest
        assert all(read_tool_result_record(m)["state"] == "recorded" for m in messages if TOOL_RESULT_RECORD_KEY in m)
        for lane in seen:
            payload = _payload(lane, messages, monkeypatch)
            seen[lane].append(payload)
            if lane in _MARKED:
                assert _marked_texts(payload) == [_expected_text(messages[expected])], (lane, number)
            else:
                assert "cache_control" not in json.dumps(payload["messages"]), lane
                assert _wire(payload) == _wire(_payload(lane, _unmarked(messages), monkeypatch)), lane
            if lane in ("anthropic", "openrouter-anthropic"):
                tools = payload["tools"]
                assert len(LLMClient._payload_cache_breakpoints(payload)) == 4
                assert ("cache_control" in tools[-1]) == (not book), "schemas get a marker only from a free slot"
    for lane, payloads in seen.items():
        for previous, current in zip(payloads, payloads[1:]):
            if lane in _MARKED:
                (before, [mark]), (after, [moved]) = _blocks(previous), _blocks(current)
                assert after[:mark + 1] == before[:mark + 1], "the last request's cached prefix is repeated"
                assert moved > mark, "the boundary advances every round"
            else:
                assert _wire(current)[:len(_wire(previous))] == _wire(previous), "byte-for-byte extension"


def test_native_replay_stays_exact_and_unmarked_while_the_boundary_follows(monkeypatch):
    from ouroboros.anthropic_native_custody import anthropic_replay_scope
    from tests.test_anthropic_native_custody import _canonical_message, _raw_content, _target, _tool_results

    target = _target()
    messages = [{"role": "user", "content": "task"}, _canonical_message(target), *_tool_results()]
    seal_task_transcript(messages, min_prefix_tokens=0)
    assert [i for i, m in enumerate(messages) if "cache_control" in m] == [3]
    client = LLMClient(api_key="unused")
    with anthropic_replay_scope():
        _system, physical = client._build_anthropic_messages(messages, target)
        client._normalize_payload_cache_ttl(target, {"messages": physical})
    assert physical[1] == {"role": "assistant", "content": _raw_content()}, "native replay stays byte-exact"
    results = physical[2]["content"]
    assert [block["tool_use_id"] for block in results] == ["tool-A", "tool-B"]
    assert results[0]["content"] == "result-A" and results[1]["content"][0]["text"] == "result-B"
    assert "cache_control" in results[1]["content"][0]
    lone = [{"role": "user", "content": "task"}, _canonical_message(target)]
    seal_task_transcript(lone, min_prefix_tokens=0)
    assert [i for i, m in enumerate(lone) if "cache_control" in m] == [0], "an assistant turn is never the anchor"


def test_an_unchanged_reprepare_reuses_its_facts_row_beside_the_mark(tmp_path, monkeypatch):
    from ouroboros import loop
    from ouroboros.loop_messages import append_context_facts
    from tests.test_loop_compaction import _ctx, _fit

    monkeypatch.setattr(loop, "_measure_round_main_fit", lambda *_args, **_kwargs: _fit())
    monkeypatch.setattr(loop, "_wrapup_global_remaining", lambda: None)
    ctx = _ctx(tmp_path)
    for round_idx in (1, 2):
        ctx.round_idx = round_idx
        seal_task_transcript(ctx.messages, min_prefix_tokens=0)
        assert append_context_facts(ctx) is True
        facts = ctx.messages[-1]
        assert append_context_facts(ctx) is False and ctx.messages[-1] is facts, "one facts row per round"
        # A free same-round redial reseals at loop entry without rewriting the history.
        seal_task_transcript(ctx.messages, min_prefix_tokens=0)
        assert append_context_facts(ctx) is False and ctx.messages[-1] is facts
        assert isinstance(facts["content"], str) and host_row_intact(facts)
        ctx.messages += [bound_host_row(CLOCK_ROW_NAME, f"[Host clock at this request: round {round_idx}]"),
                         {"role": "assistant", "content": f"answer {round_idx}"}]
    seal_task_transcript(ctx.messages, min_prefix_tokens=0)
    marked = [m for m in ctx.messages if "cache_control" in m]
    assert len(marked) == 1 and marked[0][HOST_CONTEXT_KIND_KEY] == CLOCK_ROW_NAME and host_row_intact(marked[0])
    rows = [i for i, m in enumerate(ctx.messages)
            if m.get(HOST_CONTEXT_KIND_KEY) in (CONTEXT_FACTS_NAME, CLOCK_ROW_NAME)]
    assert _obsolete_host_rows(ctx.messages) == set(rows[:2])  # the first round's facts and clock rows


def test_an_old_sealed_transcript_keeps_its_marker_free_wire_and_affinity():
    """A checkpoint written by the old in-place seal resumes: marker-free lanes send the bytes
    they sent before, the sticky session holds, and one row mark replaces the old markers."""
    marker = {"type": "ephemeral"}
    call = {"id": "t0", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
    legacy = [_system(book=False),
              {"role": "user", "content": [{"type": "text", "text": "task", "cache_control": marker}]},
              {"role": "assistant", "content": "", "tool_calls": [call]},
              {"role": "tool", "tool_call_id": "t0",
               "content": [{"type": "text", "text": "old seal", "cache_control": marker}]}]
    resumed = copy.deepcopy(legacy)
    seal_task_transcript(resumed)
    assert resumed[1]["content"] == [{"type": "text", "text": "task"}] and resumed[3]["content"] == "old seal"
    assert [i for i, m in enumerate(resumed) if "cache_control" in m] == [3]
    for lane in ("openrouter-grok", "openai", "claudexor"):
        assert _wire(_payload(lane, resumed, None)) == _wire(_payload(lane, legacy, None)), lane
    for model in ("anthropic/claude-fable-5", "x-ai/grok-4.7"):
        assert LLMClient._openrouter_session_identity(model, resumed) == LLMClient._openrouter_session_identity(
            model, legacy)
    assert _marked_texts(_payload("openrouter-anthropic", resumed, None)) == ["old seal"]


def test_user_send_copy_block_shapes_survive_boundary_moves_without_changing_callers():
    """OpenRouter may coalesce adjacent strings differently from explicit blocks.
    Preserve the representation, not just concatenated text, as the marker moves."""
    messages = [_system(False), {"role": "user", "content": "owner words"},
                {"role": "user", "content": "host facts"}]
    originals = copy.deepcopy(messages)
    snapshots = []
    for number in range(3):
        seal_task_transcript(messages)
        snapshots.append(_payload("openrouter-anthropic", messages, None))
        messages.extend([{"role": "assistant", "content": "ACK"},
                         {"role": "user", "content": f"next facts {number}"},
                         {"role": "user", "content": f"next question {number}"}])
    def without_markers(rows):
        return [{**m, "content": [{k: v for k, v in b.items() if k != "cache_control"}
                                  for b in m["content"]]}
                if isinstance(m.get("content"), list) else m for m in rows]
    for previous, current in zip(snapshots, snapshots[1:]):
        assert without_markers(current["messages"][:len(previous["messages"])]) == without_markers(previous["messages"])
    # No semantic/canonical wrapping, and no change to an independent review caller.
    assert _unmarked(messages[:len(originals)]) == originals
    unsealed = _payload("openrouter-anthropic", originals, None)
    assert unsealed["messages"][-1]["content"] == "host facts"
