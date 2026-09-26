"""Only actual Nano startup relieves measured source pressure before Main."""

import json

import pytest

from ouroboros import context
from ouroboros.context_fit import estimate_context_prompt_tokens
from ouroboros.tools.registry import ToolContext
from tests.test_doc_context import _make_env_and_memory
from tests import test_consolidator_context_fit as fit_helpers
from tests.test_memory_pressure_maintenance import SourceReader

fit = fit_helpers.fit


def _setup(tmp_path):
    env, memory = _make_env_and_memory(tmp_path)
    chat = env.drive_root / "logs/chat.jsonl"
    raw = json.dumps({"chat_id": 1, "direction": "in", "text": "Complete original discussion. " * 16000 + "FINAL OWNER DECISION",
                      "ts": "2026-09-13T00:00:00Z"}) + "\n"
    chat.write_text(raw)
    task = {"id": "startup", "type": "task", "text": "Continue with my exact final decision."}
    ctx = ToolContext(repo_dir=env.repo_dir, drive_root=env.drive_root, task_id=task["id"])
    return env, memory, task, ctx, raw


def test_actual_nano_preparation_consolidates_complete_source_before_returning(tmp_path, fit, monkeypatch):
    fit.window = 50000
    env, memory, task, ctx, raw = _setup(tmp_path)
    monkeypatch.setattr(context, "get_context_mode", lambda: "nano")
    def fits(messages, tools):
        measured = estimate_context_prompt_tokens(messages, tools)
        return {"accepted": measured < 16000, "input_tokens": measured, "strict_bound_proven": False}
    actor = SourceReader(env.drive_root, fit.window)
    before = context.build_context_fit_plan(env, memory, task, preferred_mode="nano", ctx=ctx)
    assert not fits(before.messages_for("nano"), [])["accepted"]
    messages, info = context.build_llm_messages(env, memory, task, ctx=ctx, llm=actor,
                                               tool_schemas=[], fit_candidate=fits)
    assert info["context_memory_maintenance"]["status"] == "fitting"
    assert fits(messages, [])["accepted"]
    assert (env.drive_root / "logs/chat.jsonl").read_text() == raw
    assert actor.calls and info["context_memory_maintenance"]["usage"]["cost"] > 0
    events = [json.loads(line) for line in (env.drive_root / "logs/events.jsonl").read_text().splitlines()]
    receipt = next(row for row in events if row.get("type") == "context_memory_maintenance")
    assert receipt["task_id"] == task["id"] and receipt["changed_sources"]
    assert receipt["status"] == "fitting"
    assert messages[-1]["content"] == task["text"]
    # Repeated preparation sees the consolidated source; it does not buy another maintenance run.
    calls = len(actor.calls)
    context.build_llm_messages(env, memory, task, ctx=ctx, llm=actor, tool_schemas=[], fit_candidate=fits)
    assert len(actor.calls) == calls


@pytest.mark.parametrize("broken", [b"{bad", b"[]", b'{"pending_knowledge_nominations":[],"pending_knowledge_nominations":[]}'])
def test_nano_unreadable_dialogue_meta_withholds_maintenance_not_main(tmp_path, fit, monkeypatch, broken):
    env, memory, task, ctx, chat_before = _setup(tmp_path)
    meta = env.drive_root / "memory/dialogue_meta.json"
    meta.write_bytes(broken)
    monkeypatch.setattr(context, "get_context_mode", lambda: "nano")
    actor = SourceReader(env.drive_root, fit.window)
    messages, info = context.build_llm_messages(
        env, memory, task, ctx=ctx, llm=actor, tool_schemas=[],
        fit_candidate=lambda _messages, _tools: {"accepted": False},
    )
    receipt = info["context_memory_maintenance"]
    assert receipt["status"] == "no_progress"
    assert receipt["usage"]["_consolidation_errors"][0]["kind"] == "dialogue_meta_unreadable"
    assert messages[-1]["content"] == task["text"]
    assert meta.read_bytes() == broken
    assert (env.drive_root / "logs/chat.jsonl").read_text() == chat_before
    assert not actor.calls
    events = [json.loads(line) for line in (env.drive_root / "logs/events.jsonl").read_text().splitlines()]
    assert any(row.get("type") == "context_memory_maintenance" and
               row["usage"]["_consolidation_errors"][0]["kind"] == "dialogue_meta_unreadable"
               for row in events)


def test_max_and_pure_preview_never_start_a_maintenance_model(tmp_path, fit, monkeypatch):
    env, memory, task, ctx, _raw = _setup(tmp_path)
    actor = SourceReader(env.drive_root, 50000)
    monkeypatch.setattr(context, "get_context_mode", lambda: "max")
    context.build_llm_messages(env, memory, task, ctx=ctx, llm=actor,
                               tool_schemas=[], fit_candidate=lambda m,t: {"accepted": False})
    monkeypatch.setattr(context, "get_context_mode", lambda: "nano")
    context.build_context_fit_plan(env, memory, task, preferred_mode="nano", ctx=ctx)
    assert not actor.calls
