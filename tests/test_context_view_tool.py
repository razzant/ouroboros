"""The existing compact tool pins an observed view and applies after its own pair."""

import copy
import json
from types import SimpleNamespace

import pytest

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
    notice = json.loads(receipts[0]["content"].split("\n", 1)[1])  # what the fold freed and kept (C2 S3)
    assert notice["reclaimed_tokens"] == ctx._context_view_receipt["reclaimed_tokens"] and notice["basis"] == "host estimates"
    from ouroboros.tools.compact_context import residue_by_kind
    assert notice["residue"] == residue_by_kind(candidate[:candidate.index(receipts[0])])  # the view it kept
    assert notice["residue"]["capsule"]["units"] == 1 and notice["residue"]["tool"]["tokens"] > 0
    assert "resulting_input_tokens" in notice
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


@pytest.mark.parametrize("outcome", ["applied", "fit_rejected", "retention_error"])
def test_checkpoint_event_keeps_final_facts_and_resolves_saved_structures(tmp_path, monkeypatch, outcome):
    from ouroboros import artifacts, loop
    from ouroboros.loop_round_limits import _run_authored_context_view
    from ouroboros.tools.chronicle import _memory_read

    ctx = SimpleNamespace(active_context_mode="low")
    messages, schemas, events = _source(), [], []
    record_context_view(ctx, messages, schemas)
    _compact_context(ctx, working_note="" if outcome == "no_op" else "I retain the conclusion.")

    def fit(candidate, tools):
        last = candidate[-1].get("content")
        final = isinstance(last, str) and last.startswith("[Context view receipt]")
        return {"accepted": not (outcome == "fit_rejected" and final), "measured_final": final}

    frame = _context(tmp_path, ctx, schemas, monkeypatch, fit)
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda q, t, logs, event: events.append(copy.deepcopy(event)))
    store = artifacts.store_actor_source_bytes

    def save(*args, **kwargs):
        if outcome == "store_error" and json.loads(kwargs["data"]).get("stage") == "candidate_materialization":
            raise OSError("receipt storage unavailable")
        return store(*args, **kwargs)

    monkeypatch.setattr(artifacts, "store_actor_source_bytes", save)
    if outcome == "retention_error":
        from ouroboros import observability

        def unavailable(*args, **kwargs):
            raise OSError("canonical source storage unavailable")

        monkeypatch.setattr(observability, "promote_task_source_ref", unavailable)
    candidate = _run_authored_context_view(messages, frame, ctx._pending_compaction, None)
    event, = events
    receipt = ctx._context_view_receipt
    assert event["status"] == receipt["status"] == ("applied" if outcome.endswith("error") else outcome)
    assert event["checkpoint_kind"] == "context_view" and event["round"] == 1
    if outcome in ("no_op", "store_error", "retention_error"):
        assert "receipt_source" not in event and "at_source" not in event
        assert event == {"checkpoint_kind": "context_view", "round": 1, **receipt}
        if outcome == "no_op":
            assert candidate is messages
        elif outcome == "store_error":
            assert '"receipt_source": "unavailable"' in candidate[-1]["content"]
        else:
            notice = json.loads(candidate[-1]["content"].split("\n", 1)[1])
            assert notice["receipt_source"]["read"]["tool"] == "read_file"
        return

    source = event["receipt_source"]
    assert source["task_id"] == "view-task" and source["read"]["tool"] == "memory_read"
    reader = SimpleNamespace(task_id="another-reader", drive_root=tmp_path, budget_drive_root=tmp_path)
    text = _memory_read(reader, **source["read"]["arguments"])
    saved = json.loads(text.split("\n", 2)[2])
    assert saved["stage"] == event["receipt_stage"] == "candidate_materialization"
    assert saved["published"] is False and saved["receipt"]["status"] == "applied"
    assert event["at_source"]["source_refs"] == len(saved["receipt"]["source_refs"]) > 0
    for key, count in event["at_source"].items():
        assert key not in event
        assert saved["receipt"][key] == json.loads(json.dumps(receipt[key]))
        assert count == len(saved["receipt"][key])
    if outcome == "fit_rejected":
        assert candidate is messages and event["fit"] == {"accepted": False, "measured_final": True}
    else:
        assert event["view_revision"] == event["after_transcript_sha256"] == receipt["view_revision"]
        assert event["fit_after"] == {"accepted": True, "measured_final": True}
        model_notice = json.loads(candidate[-1]["content"].split("\n", 1)[1])
        assert "task_id" not in model_notice["receipt_source"]
        assert model_notice["receipt_source"]["read"]["tool"] == "read_file"


@pytest.mark.parametrize("accepted", [True, False])
def test_checkpoint_event_still_explains_its_structures_after_child_gc(tmp_path, monkeypatch, accepted):
    from ouroboros import headless, loop, observability
    from ouroboros.loop_round_limits import _run_authored_context_view
    from ouroboros.task_custody import settle_child_drive
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.chronicle import _memory_read
    from ouroboros.tools.compact_context import _compact_context, record_context_view

    parent = tmp_path / "parent"
    task_id = "view-task"
    child = parent / "task_drives" / task_id
    child.mkdir(parents=True)
    write_task_result(parent, task_id, "running", root_task_id="parent")
    ctx = SimpleNamespace(active_context_mode="low")
    messages, schemas, events = _source(), [], []
    observability.persist_call(child, task_id=task_id, call_id="before", call_type="llm_request",
                               payload={"messages": messages})
    record_context_view(ctx, messages, schemas)
    _compact_context(ctx, working_note="I retain the conclusion.")

    def fit(candidate, tools):
        last = candidate[-1].get("content")
        final = isinstance(last, str) and last.startswith("[Context view receipt]")
        return {"accepted": accepted or not final, "measured_final": final}

    frame = _context(child, ctx, schemas, monkeypatch, fit)
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda q, t, logs, event: events.append(copy.deepcopy(event)))
    candidate = _run_authored_context_view(messages, frame, ctx._pending_compaction, None)
    event, = events
    assert "receipt_source" in event
    assert event["status"] == ("applied" if accepted else "fit_rejected")
    reader = SimpleNamespace(task_id="another-reader", drive_root=parent, budget_drive_root=parent)

    def structures():
        if "receipt_source" not in event:
            return event
        text = _memory_read(reader, **event["receipt_source"]["read"]["arguments"])
        try:
            return json.loads(text.split("\n", 2)[2])["receipt"]
        except (ValueError, IndexError, KeyError):
            return {"read_error": text}

    before = structures()
    assert before["source_refs"]
    observability.persist_call(child, task_id=task_id, call_id="after", call_type="llm_request",
                               payload={"messages": candidate})
    write_task_result(child, task_id, "completed", result="The answer", artifact_status="ready")
    headless.copy_child_task_result(parent, {"id": task_id, "drive_root": str(child)})
    headless.retry_child_task_refs(parent, child, task_id)
    outcome = settle_child_drive(parent, task_id, child, live=lambda _: False)
    assert outcome["status"] == "removed", outcome
    assert not child.exists()
    from ouroboros.artifacts import read_actor_source_bytes
    retained = read_actor_source_bytes(parent, task_id, event["receipt_source"])
    assert json.loads(retained)["receipt"] == before
    after = structures()
    assert after.get("source_refs") == before["source_refs"], after
