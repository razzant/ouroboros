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


def _read(call_id):
    return [{"role": "assistant", "tool_calls": [{"id": call_id, "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": call_id, "content": f"Exact evidence {call_id} " * 100}]


def _inspected(ctx, kind):
    return [row["unit_id"] for row in json.loads(_compact_context(ctx, inspect=True))["units"] if row["kind"] == kind]


def test_keep_last_n_keeps_unnamed_notes_and_never_re_protects_a_named_one(tmp_path, monkeypatch):
    """Unchanged for unnamed notes; the count's derived keep set does not protect a note the mind
    names, not even one inside the count's recent window."""
    from tests.test_compaction import _note

    older, note = _note(tmp_path, "Goal: keep the owner's exact wording.", "older"), _note(tmp_path, "r2 checked.")
    messages = [*_source()[:2], older, *_read("r1"), *_read("r2"), note, *_read("r3")]
    ctx = SimpleNamespace(active_context_mode="low")
    record_context_view(ctx, messages, [])
    older_id, note_id = _inspected(ctx, "capsule")
    assert "requested" in _compact_context(ctx, working_note="r1 agreed with the goal.", keep_last_n=2)
    kept = _run_round_compaction(copy.deepcopy(messages), _context(tmp_path, ctx, [], monkeypatch))
    assert ctx._context_view_receipt["status"] == "applied" and older in kept and note in kept
    assert [m["tool_call_id"] for m in kept if m.get("role") == "tool"] == ["r2", "r3"]

    record_context_view(ctx, messages, [])
    assert "requested" in _compact_context(ctx, working_note="r1 and the r2 check, merged.", keep_last_n=2,
                                           replace_note_ids=[note_id])
    merged = _run_round_compaction(copy.deepcopy(messages), _context(tmp_path, ctx, [], monkeypatch))
    receipt = ctx._context_view_receipt
    assert receipt["status"] == "applied" and older in merged and note not in merged
    assert [m["tool_call_id"] for m in merged if m.get("role") == "tool"] == ["r2", "r3"]
    assert [row["unit_id"] for row in receipt["replaced_notes"]["notes"]] == [note_id]


def test_inspect_identifies_capsule_authorship_and_first_line_for_a_named_merge(tmp_path, monkeypatch):
    ctx = SimpleNamespace(active_context_mode="low")
    source = _source()
    record_context_view(ctx, source, [])
    _compact_context(ctx, working_note="Goal: keep the original wording.\nDetails stay in the note.", keep_unit_ids=[])
    folded = _run_round_compaction(source, _context(tmp_path, ctx, [], monkeypatch))
    note = folded[2]
    helper = copy.deepcopy(note)
    helper["content"][0]["_context_capsule"]["authorship"] = "helper"
    legacy = copy.deepcopy(note)
    del legacy["content"][0]["_context_capsule"]["authorship"]
    messages = [*source[:2], note, helper, legacy, *_read("next")]
    record_context_view(ctx, messages, [])
    seen = json.loads(_compact_context(ctx, inspect=True))
    capsules = [row for row in seen["units"] if row["kind"] == "capsule"]
    assert [row.get("authorship") for row in capsules] == ["actor", "helper", "unrecorded"]
    assert [row["first_line"] for row in capsules] == ["Goal: keep the original wording."] * 3
    assert all("first_line" not in row and "authorship" not in row
               for row in seen["units"] if row["kind"] != "capsule")
    _compact_context(ctx, working_note="The helper's account is incorporated.", keep_unit_ids=[],
                     replace_note_ids=[capsules[1]["unit_id"]])
    merged = _run_round_compaction(messages, _context(tmp_path, ctx, [], monkeypatch))
    assert helper not in merged and note in merged and legacy in merged


def test_keeping_and_replacing_one_note_or_naming_a_non_note_is_an_argument_error(tmp_path, monkeypatch):
    from tests.test_compaction import _note

    note = _note(tmp_path, "An earlier note.")
    messages = [*_source(), note]
    ctx = SimpleNamespace(active_context_mode="low", _active_builtin_tool_result=None)
    record_context_view(ctx, messages, [])
    note_id, = _inspected(ctx, "capsule")
    tool_id, = _inspected(ctx, "tool")
    for arguments, named in (({"keep_unit_ids": [note_id], "replace_note_ids": [note_id]}, note_id),
                             ({"replace_note_ids": [tool_id]}, tool_id),
                             ({"keep_last_n": 2, "replace_note_ids": ["unit:9:9:unseen"]}, "unit:9:9:unseen")):
        text = _compact_context(ctx, working_note="Merged.", **arguments)
        assert named in text and text.endswith("Current context is unchanged.")
        assert ctx._active_builtin_tool_result.code == "TOOL_ARG_ERROR"
        assert getattr(ctx, "_pending_compaction", None) is None
    assert "requested" in _compact_context(ctx, working_note="Merged.", replace_note_ids=[note_id])
    result = _run_round_compaction(copy.deepcopy(messages), _context(tmp_path, ctx, [], monkeypatch))
    assert ctx._context_view_receipt["status"] == "applied" and note not in result


@pytest.mark.parametrize("saved", ["before_replace_note_ids", "naming_the_note"])
def test_a_request_restored_after_a_pause_without_replace_note_ids_names_no_note(tmp_path, monkeypatch, saved):
    """A pending request an earlier release saved has no replace_note_ids: it names none and folds as
    before minus the notes; a request saved with it still names its note after the JSON continuation."""
    from ouroboros.owner_wait import continuation_state, restore_continuation_state
    from tests.test_compaction import _note

    note, other = _note(tmp_path, "Goal: keep the owner's exact wording."), _note(tmp_path, "Another note.", "other")
    messages, schemas = [*_source(), note, other], []
    ctx = SimpleNamespace(task_id="paused-view", drive_root=tmp_path, budget_drive_root=tmp_path,
                          active_context_mode="low")
    record_context_view(ctx, messages, schemas)
    named = {"replace_note_ids": _inspected(ctx, "capsule")[:1]} if saved == "naming_the_note" else {}
    _compact_context(ctx, working_note="The read is understood.", keep_unit_ids=[], **named)
    state = json.loads(json.dumps(continuation_state(ctx, messages, {}, {}, 1, schemas, set())))
    if saved == "before_replace_note_ids":
        del state["context_observations"]["_pending_compaction"]["replace_note_ids"]
    resumed = SimpleNamespace(task_id=ctx.task_id, drive_root=tmp_path, budget_drive_root=tmp_path,
                              active_context_mode="low")
    restored = []
    restore_continuation_state(SimpleNamespace(_ctx=resumed), state, restored, {}, {}, set())
    assert restored == messages  # the saved view, its capsule included, comes back exactly
    result = _run_round_compaction(restored, _context(tmp_path, resumed, schemas, monkeypatch))
    assert resumed._context_view_receipt["status"] == "applied"
    assert not any(m.get("tool_call_id") == "read" for m in result)
    assert other in result and (note in result) is (saved == "before_replace_note_ids")
