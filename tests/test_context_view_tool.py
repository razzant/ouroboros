"""The existing compact tool pins an observed view and applies after its own pair."""

import copy
import json
from types import SimpleNamespace

from ouroboros.tools.compact_context import _compact_context, record_context_view
from ouroboros.loop_round_limits import _CompactionRoundContext, _run_round_compaction


def _source():
    return [{"role": "system", "content": "Full unchanged identity/books"},
            {"role": "user", "content": "Owner task"},
            {"role": "assistant", "tool_calls": [{"id": "read", "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "read", "content": "Long exact evidence " * 200}]


def _context(tmp_path, ctx, schemas, monkeypatch, fit=lambda m,t: {"accepted": True}):
    from ouroboros import loop
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda *a, **k: None)
    registry = SimpleNamespace(_ctx=ctx, schemas=lambda: schemas)
    return _CompactionRoundContext(registry, tmp_path, tmp_path / "logs", "view-task", 1,
                                   None, lambda text: None, schemas, fit)


def test_inspect_then_new_model_turn_and_owner_tail_preserves_pinned_source(tmp_path, monkeypatch):
    ctx = SimpleNamespace(active_context_mode="low")
    messages, schemas = _source(), []
    record_context_view(ctx, messages, schemas)
    seen = json.loads(_compact_context(ctx, inspect=True))
    tail = [{"role": "user", "content": "New exact owner correction"}]
    current = copy.deepcopy(messages) + tail
    record_context_view(ctx, current, schemas)
    text = _compact_context(ctx, expected_view_revision=seen["view_revision"],
                            working_note="I learned from that evidence.", keep_unit_ids=[])
    assert "requested" in text
    current += [{"role": "assistant", "tool_calls": [{"id": "compact", "function": {
        "name": "compact_context", "arguments": "{}"}}]}, {"role": "tool", "tool_call_id": "compact", "content": text}]
    candidate = _run_round_compaction(current, _context(tmp_path, ctx, schemas, monkeypatch))
    assert ctx._context_view_receipt["status"] == "applied"
    assert candidate[0] == messages[0]
    assert tail[0] in candidate
    assert any(m.get("tool_call_id") == "compact" for m in candidate)
    receipts = [m for m in candidate if isinstance(m.get("content"), str)
                and m["content"].startswith("[Context view receipt]")]
    assert len(receipts) == 1 and receipts[0]["role"] == "user"
    assert "[INDEPENDENT_ROOTS]" not in receipts[0]["content"]
    assert candidate.index(receipts[0]) > next(
        i for i, m in enumerate(candidate) if m.get("tool_call_id") == "compact")
    assert ctx._pending_compaction is None


def test_noop_retains_exact_messages_schema_and_cache_state(tmp_path, monkeypatch):
    from ouroboros import loop_round_limits
    ctx = SimpleNamespace(active_context_mode="low", model_turn_state=object())
    messages, schemas = _source(), []
    turn = ctx.model_turn_state
    record_context_view(ctx, messages, schemas)
    seen = json.loads(_compact_context(ctx, inspect=True))
    _compact_context(ctx, expected_view_revision=seen["view_revision"], working_note="")
    monkeypatch.setattr(loop_round_limits, "invalidate_task_cache_splits", lambda *_: (_ for _ in ()).throw(AssertionError("no-op invalidation")))
    result = _run_round_compaction(messages, _context(tmp_path, ctx, schemas, monkeypatch))
    assert result is messages and ctx._context_view_receipt["status"] == "no_op"
    assert ctx.model_turn_state is turn


def test_fit_refusal_does_not_publish_schemas_or_replace_source(tmp_path, monkeypatch):
    ctx = SimpleNamespace(active_context_mode="low")
    messages, schemas = _source(), []
    record_context_view(ctx, messages, schemas)
    seen = json.loads(_compact_context(ctx, inspect=True))
    _compact_context(ctx, expected_view_revision=seen["view_revision"], working_note="A note", keep_unit_ids=[])
    result = _run_round_compaction(messages, _context(tmp_path, ctx, schemas, monkeypatch,
                                                        lambda m,t: {"accepted": False, "reason": "unfit"}))
    assert result == messages and not schemas
    assert ctx._context_view_receipt["status"] == "fit_rejected"


def test_receipt_itself_is_included_in_final_fit_before_publication(tmp_path, monkeypatch):
    ctx = SimpleNamespace(active_context_mode="low")
    messages, schemas = _source(), []
    record_context_view(ctx, messages, schemas)
    seen = json.loads(_compact_context(ctx, inspect=True))
    _compact_context(ctx, expected_view_revision=seen["view_revision"], working_note="A note", keep_unit_ids=[])
    calls = []
    def fit(candidate, tools):
        calls.append(candidate)
        return {"accepted": not candidate[-1].get("content", "").startswith("[Context view receipt]")
                if isinstance(candidate[-1].get("content"), str) else True}
    result = _run_round_compaction(messages, _context(tmp_path, ctx, schemas, monkeypatch, fit))
    assert len(calls) == 2 and result is messages
    assert ctx._context_view_receipt["status"] == "fit_rejected"


def test_omitted_revision_binds_this_actors_actual_send_without_inspection(tmp_path, monkeypatch):
    ctx, other = SimpleNamespace(active_context_mode="low"), SimpleNamespace(active_context_mode="low")
    messages, schemas = _source(), []
    record_context_view(ctx, messages, schemas)
    record_context_view(other, [{"role": "user", "content": "Another actor"}], schemas)
    response = _compact_context(ctx, working_note="I retained the evidence and will continue.", keep_unit_ids=[])
    assert "requested" in response
    current = [*messages, {"role": "user", "content": "Owner arrived after that send"}]
    result = _run_round_compaction(current, _context(tmp_path, ctx, schemas, monkeypatch))
    assert ctx._context_view_receipt["status"] == "applied"
    assert current[-1] in result
    assert other._last_context_observation["messages"][0]["content"] == "Another actor"
    assert _compact_context(ctx, expected_view_revision="unrelated", working_note="stale", keep_unit_ids=[]).startswith("Context view mismatch")
