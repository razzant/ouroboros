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
    # The candidate, then with its receipt, whose lists already stand behind its address as counts: a receipt that
    # fits no window undoes the view, and no list is dropped on the way.
    assert len(calls) == 2 and result is messages
    assert '"at_source": {"bodies_removed": 1, "notes_kept": 1}' in calls[1][-1]["content"]
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
            assert set(notice["receipt_source"]) == {"kind", "root", "path", "size", "sha256"}
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
        assert set(model_notice["receipt_source"]) == {"kind", "root", "path", "size", "sha256"}  # read_file's address


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
    from tests._working_notes import _note

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
    from tests._working_notes import _note

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
    from tests._working_notes import _note

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


@pytest.mark.parametrize("saved", ["with_its_write_time", "before_notes_carried_one"])
def test_a_note_is_dated_by_its_compact_context_call_even_when_a_pause_applies_it_later(tmp_path, monkeypatch, saved):
    """Both directions: the note carries the time the mind wrote it (its compact_context call), even when the view is
    applied after a pause saved the request; a request saved before requests carried that time is dated when applied."""
    import datetime
    from ouroboros import utils
    from ouroboros.context_compaction import _capsule_metadata, note_written
    from ouroboros.owner_wait import continuation_state, restore_continuation_state

    messages, schemas = _source(), []
    ctx = SimpleNamespace(task_id="paused-view", drive_root=tmp_path, budget_drive_root=tmp_path,
                          active_context_mode="low")
    record_context_view(ctx, messages, schemas)
    monkeypatch.setattr(utils, "utc_now_iso", lambda timespec="auto": "2026-11-13T19:48:59+00:00")
    _compact_context(ctx, working_note="The read is understood.", keep_unit_ids=[])
    monkeypatch.undo()
    state = json.loads(json.dumps(continuation_state(ctx, messages, {}, {}, 1, schemas, set())))
    if saved == "before_notes_carried_one":
        del state["context_observations"]["_pending_compaction"]["written_at"]
    resumed = SimpleNamespace(task_id=ctx.task_id, drive_root=tmp_path, budget_drive_root=tmp_path,
                              active_context_mode="low")
    restored = []
    restore_continuation_state(SimpleNamespace(_ctx=resumed), state, restored, {}, {}, set())

    result = _run_round_compaction(restored, _context(tmp_path, resumed, schemas, monkeypatch))
    meta = next(meta for row in result if (meta := _capsule_metadata(row)[1]))
    if saved == "with_its_write_time":
        assert meta["written_at"] == "2026-11-13T19:48:59+00:00" and note_written(meta) == "2026-11-13 19:48 UTC"
    else:
        applied = datetime.datetime.fromisoformat(meta["written_at"])
        assert abs(datetime.datetime.now(datetime.timezone.utc) - applied) < datetime.timedelta(minutes=5)
    assert next(row for row in result if _capsule_metadata(row)[1])["content"][0]["text"].startswith(
        f"[Actor-authored working view, written {note_written(meta)}; ")


def _receipt(rows):
    """The view's receipt row and the view it closes; host producers may append rows after it."""
    index = next(i for i, row in enumerate(rows) if str(row.get("content")).startswith("[Context view receipt]"))
    return json.loads(rows[index]["content"].split("\n", 1)[1]), rows[:index]


def test_a_fold_receipt_counts_its_notes_and_bodies_and_keeps_their_lists_at_its_address(tmp_path, monkeypatch):
    """Both directions: an applied fold's receipt row counts the notes in view and the bodies that left
    (``at_source``) beside the exact address of the full receipt, and lists neither; the stored receipt there lists
    every note (id, write time or "not recorded", first line) and every body that left (what it was, its address),
    and that address restores the body; a restore that removes nothing counts no body."""
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.context_compaction import context_units
    from tests._working_notes import _note, _undated

    older = _undated(_note(tmp_path, "Goal: keep the owner's exact wording.", "older"))
    messages = [*_source(), older]
    ctx = SimpleNamespace(active_context_mode="low")
    record_context_view(ctx, messages, [])
    _compact_context(ctx, working_note="The read confirms the goal.", keep_unit_ids=[])
    folded = _run_round_compaction(copy.deepcopy(messages), _context(tmp_path, ctx, [], monkeypatch))
    notice, view = _receipt(folded)
    assert notice["at_source"] == {"bodies_removed": 1, "notes_kept": 2}
    assert not {"bodies_removed", "notes_kept", "restore_checkpoint"} & set(notice)
    stored = json.loads(read_actor_source_bytes(tmp_path, "view-task", notice["receipt_source"]))["receipt"]
    capsule_ids = [unit.unit_id for unit in context_units(view) if unit.start >= 2]
    assert [set(note) for note in stored["notes_kept"]] == [{"id", "written", "first_line"}] * 2
    assert [note["id"] for note in stored["notes_kept"]] == capsule_ids
    assert [note["first_line"] for note in stored["notes_kept"]] == ["The read confirms the goal.",
                                                                     "Goal: keep the owner's exact wording."]
    assert stored["notes_kept"][1]["written"] == "not recorded" and stored["notes_kept"][0]["written"].endswith(" UTC")
    read = context_units(messages, scope="dialogue")[0]
    assert stored["bodies_removed"] == [{"source": "tool read_file {}",
                                         "address": {"unit_id": read.unit_id, "raw_sha256": read.raw_sha256}}]
    assert set(stored["restore_checkpoint"]) == {"kind", "root", "path", "size", "sha256"}

    record_context_view(ctx, view, [])
    ref = {"checkpoint_ref": stored["restore_checkpoint"], **stored["bodies_removed"][0]["address"]}
    assert "requested" in _compact_context(ctx, restore_unit_refs=[ref])
    restored = _run_round_compaction(copy.deepcopy(view), _context(tmp_path, ctx, [], monkeypatch))
    again, _ = _receipt(restored)
    assert ctx._context_view_receipt["status"] == "applied" and "Long exact evidence" in json.dumps(restored)
    assert again["at_source"] == {"notes_kept": 2} and len(ctx._context_view_receipt["notes_kept"]) == 2


def test_a_full_folds_receipt_row_is_one_size_class_for_five_bodies_and_a_hundred(tmp_path, monkeypatch):
    """The scale invariant: the receipt row of a full fold that removed 100 bodies is the size of one that removed 5
    (only its counts differ), and with one note in view it stays within about 350 tokens (1,050 characters at three
    characters a token; N6 measured 4.9k tokens at the median while it listed every body)."""
    rows = {}
    for count in (5, 100):
        messages = [*_source()[:2]]
        for index in range(count):
            messages += [{"role": "assistant", "tool_calls": [{"id": f"r{index}", "function": {
                "name": "read_file", "arguments": json.dumps({"path": f"src/module_{index}.py"})}}]},
                {"role": "tool", "tool_call_id": f"r{index}", "content": f"evidence {index} " * 40}]
        ctx = SimpleNamespace(active_context_mode="low")
        record_context_view(ctx, messages, [])
        _compact_context(ctx, working_note="One note over every read.", keep_unit_ids=[])
        folded = _run_round_compaction(messages, _context(tmp_path / str(count), ctx, [], monkeypatch))
        row = next(row["content"] for row in folded if str(row.get("content")).startswith("[Context view receipt]"))
        rows[count] = row
        assert json.loads(row.split("\n", 1)[1])["at_source"] == {"bodies_removed": count, "notes_kept": 1}
        assert len(ctx._context_view_receipt["bodies_removed"]) == count  # the stored receipt lists them all
    assert abs(len(rows[100]) - len(rows[5])) <= 40, rows
    assert len(rows[100]) <= 1_050, rows[100]


def test_inspect_sends_its_shared_fields_on_page_one_only_and_every_unit_once(tmp_path, monkeypatch):
    """Both directions: page one carries the fields every page shares; a later page carries only its units, the paging,
    the view revision, the checkpoint its units restore from and the retained ordinary-memory map, within the page
    limit; together the pages list every unit exactly once."""
    from ouroboros import tool_capabilities

    messages = [*_source()[:2], *(row for index in range(80) for row in _read(f"r{index}"))]
    ctx = SimpleNamespace(active_context_mode="low")
    record_context_view(ctx, messages, [])
    monkeypatch.setattr(tool_capabilities, "tool_result_limit", lambda _name: 6000)
    pages, offset = [], 0
    while offset is not None:
        text = _compact_context(ctx, inspect=True, offset=offset)
        pages.append((text, json.loads(text)))
        offset = pages[-1][1].get("next_offset")

    first, later = pages[0][1], pages[1:]
    assert len(later) >= 2 and {"rule", "schema_names", "inventory", "fit_measurement", "restore_checkpoint"} <= set(first)
    for text, page in later:
        assert set(page) <= {"view_revision", "restore_checkpoint", "ordinary_memory", "offset", "total", "units",
                             "next_offset"} and page["units"]
        assert page["view_revision"] == first["view_revision"] and len(text) <= 6000
        assert page["restore_checkpoint"] == first["restore_checkpoint"]
    listed = [row["unit_id"] for _, page in pages for row in page["units"]]
    assert len(listed) == len(set(listed)) == first["total"] == first["inventory"]["units"]


def test_a_later_inspect_page_names_the_checkpoint_its_units_restore_from(tmp_path, monkeypatch):
    """Both directions: a later page read from a newer observation carries that observation's restore_checkpoint, and a
    unit only that observation holds restores from the page alone; page one's checkpoint of the earlier observation
    does not hold it. The large shared fields still come on page one only."""
    from ouroboros import tool_capabilities

    monkeypatch.setattr(tool_capabilities, "tool_result_limit", lambda _name: 6000)
    ctx = SimpleNamespace(active_context_mode="low", drive_root=tmp_path, budget_drive_root=tmp_path,
                          task_id="view-task")  # the round context `_context` builds runs as this task
    earlier = [*_source()[:2], *(row for index in range(55) for row in _read(f"r{index}"))]
    record_context_view(ctx, earlier, [])
    first = json.loads(_compact_context(ctx, inspect=True))
    assert first.get("next_offset") and first.get("restore_checkpoint")
    newer = [*earlier, {"role": "assistant", "tool_calls": [{"id": "inspect", "function": {
        "name": "compact_context", "arguments": '{"inspect": true}'}}]},
        {"role": "tool", "tool_call_id": "inspect", "content": json.dumps(first)}, *_read("newest")]
    record_context_view(ctx, newer, [])
    page = json.loads(_compact_context(ctx, inspect=True, offset=first["total"] + 1))
    assert page["view_revision"] != first["view_revision"] and page.get("restore_checkpoint")
    assert {"fit_measurement", "rule", "inventory", "schema_names", "work_room"}.isdisjoint(page)
    unit = page["units"][-1]

    def restore(checkpoint):
        _compact_context(ctx, restore_unit_refs=[{"checkpoint_ref": checkpoint, "unit_id": unit["unit_id"],
                                                  "raw_sha256": unit["raw_sha256"]}])
        _run_round_compaction(copy.deepcopy(newer), _context(tmp_path, ctx, [], monkeypatch))
        return ctx._context_view_receipt["status"]

    assert restore(first["restore_checkpoint"]) == "source_unavailable"
    assert restore(page["restore_checkpoint"]) == "applied"



@pytest.mark.parametrize("transfer", ["owed", "missing", "retained"])
def test_a_receipt_transfer_that_does_not_complete_keeps_the_round_and_names_its_error(tmp_path, monkeypatch, transfer):
    """The prologue's transfer of an applied fold's receipt may not complete: its walk left sources owed (an
    ``OSError`` naming the first, here one whose bytes cannot be located now) or found one unavailable (an
    ``ObservabilityPromotionSourceError``). Neither raise says that every failure was temporary, so the view
    names the error neutrally, its type and text, and the round goes on with the local receipt; a completed
    transfer names none. Either way the transfer's time rides the next round's receipt
    (``prologue.source_retention``)."""
    from ouroboros import artifacts, loop, observability
    from ouroboros.loop_round_limits import _run_authored_context_view

    ctx = SimpleNamespace(active_context_mode="low")
    messages, schemas, events = _source(), [], []
    record_context_view(ctx, messages, schemas)
    _compact_context(ctx, working_note="I retain the conclusion.")
    frame = _context(tmp_path, ctx, schemas, monkeypatch)
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda q, t, logs, event: events.append(copy.deepcopy(event)))
    error = {"owed": OSError("EvidenceUnavailable: evidence_unavailable: the packs catalog cannot answer now"),
             "missing": observability.ObservabilityPromotionSourceError("source_missing", "source_missing")}.get(transfer)

    def transfer_receipt(_child, _canonical, ref, *, task_id):
        if error is not None:
            raise error
        return ref

    monkeypatch.setattr(observability, "promote_task_source_ref", transfer_receipt)
    candidate = _run_authored_context_view(messages, frame, ctx._pending_compaction, None)
    event, = events
    assert ctx._context_view_receipt["status"] == event["status"] == "applied"
    notice = json.loads(candidate[-1]["content"].split("\n", 1)[1])
    assert artifacts.read_actor_source_bytes(tmp_path, "view-task", notice["receipt_source"])  # the local receipt stays readable
    if error is None:
        assert "receipt_retention_error" not in notice and "receipt_source" in event
    else:
        assert notice["receipt_retention_error"] == f"{type(error).__name__}: {error}"
        assert "receipt_source" not in event  # the event keeps every structure itself
    assert "source_retention" in ctx._round_prologue["sections_ms"], ctx._round_prologue


# --- cheap folds: what stands through a fold, the own call's pointer, the tail selector ------------------------------

def _read(call_id, text="Long exact evidence "):
    return [{"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function", "function": {
        "name": "read_file", "arguments": json.dumps({"path": f"{call_id}.py"})}}]},
        {"role": "tool", "tool_call_id": call_id, "content": text * 200}]


def _facts(text):
    from ouroboros.context_budget import bound_host_row
    from ouroboros.loop_messages import CONTEXT_FACTS_NAME

    return bound_host_row(CONTEXT_FACTS_NAME, "[CONTEXT_FACTS] " + text)


def _fold(tmp_path, monkeypatch, ctx, view, call_id, *, arrivals=(), own=None, **selector):
    """One fold the way Main applies it: the mind's call on the view it saw, its result, the boundary."""
    record_context_view(ctx, view, [])
    arguments = {"working_note": f"Note written by {call_id}.", **selector}
    response = _compact_context(ctx, **arguments)
    assert "requested" in response, response
    call_row = own or {"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function",
                       "function": {"name": "compact_context", "arguments": json.dumps(arguments)}}]}
    current = [*copy.deepcopy(view), call_row, {"role": "tool", "tool_call_id": call_id, "content": response},
               *arrivals]
    folded = _run_round_compaction(current, _context(tmp_path / call_id, ctx, [], monkeypatch))
    assert ctx._context_view_receipt["status"] == "applied", ctx._context_view_receipt
    return folded


def _notes(rows):
    from ouroboros.context_compaction import _capsule_metadata

    return [i for i, row in enumerate(rows) if _capsule_metadata(row)[1]]


def _unit_at(rows, index):
    from ouroboros.context_compaction import context_units

    return next(unit for unit in context_units(rows, scope="dialogue") if unit.start == index)


def _start():
    return [*_source()[:2], _facts("round 1")]


def test_a_full_fold_stands_after_the_start_and_releasing_its_note_or_omitting_it_takes_its_rows(tmp_path, monkeypatch):
    """Tests 1 and 2 of the order at the tool's own seam. A full fold keeps the start's host rows and writes its
    note after them, and its own call names the note; the next full fold keeps that note's whole request (call,
    result, receipt, the next facts row) and writes after it. Both directions: replacing the note takes its rows in
    the same fold, and an explicit keep_unit_ids list that omits a standing unit removes it."""
    from ouroboros.context_compaction import _capsule_metadata
    from ouroboros.tools.compact_context import NOTE_POINTER, standing_rows

    ctx = SimpleNamespace(active_context_mode="low")
    start = _start()
    first = _fold(tmp_path, monkeypatch, ctx, [*start, *_read("one")], "fold-1", keep_unit_ids=[])
    at, = _notes(first)
    assert first[:at] == start and at == len(start)
    meta = _capsule_metadata(first[at])[1]
    assert first[at]["content"][0]["text"].endswith("\nNote written by fold-1.")
    assert json.loads(first[at + 1]["tool_calls"][0]["function"]["arguments"]) == {
        "working_note": NOTE_POINTER.format(unit_id=meta["unit_id"]), "keep_unit_ids": []}
    request = [*first, _facts("round 3")]  # the request sent after that fold, up to the model's answer
    assert sorted(standing_rows(request)) == [2, *range(at + 1, len(request))]

    second = _fold(tmp_path, monkeypatch, ctx, [*request, *_read("two")], "fold-2", keep_unit_ids=[])
    assert second[:len(request)] == request and _notes(second) == [at, len(request)]

    note_id = _unit_at(request, at).unit_id
    released = _fold(tmp_path, monkeypatch, ctx, [*request, *_read("two")], "fold-3", keep_unit_ids=[],
                     replace_note_ids=[note_id])
    assert released[:at] == start and _notes(released) == [at]  # its rows went with it
    assert not any(row.get("tool_call_id") == "fold-1" for row in released)
    alone = _fold(tmp_path, monkeypatch, ctx, [*request, *_read("two")], "fold-5", replace_note_ids=[note_id])
    assert not any(row.get("tool_call_id") == "fold-1" for row in alone)  # omitted selectors: only its own rows go
    assert alone[:at] == start and any(row.get("tool_call_id") == "two" for row in alone)
    omitted = _fold(tmp_path, monkeypatch, ctx, [*request, *_read("two")], "fold-4",
                    keep_unit_ids=[_unit_at(request, at + 1).unit_id])
    assert request[2] not in omitted and request[at + 1] in omitted  # the mind's explicit choice still decides


def test_keep_last_n_leaves_no_anchor_and_the_next_full_fold_takes_its_tail(tmp_path, monkeypatch):
    """Test 3: a keep_last_n fold keeps an earlier full fold's standing rows untouched, writes its note after them
    and leaves no anchor of its own (its call keeps its argument); the next full fold folds its kept tail and its
    own rows, and writes its note right after the keep_last_n fold's note."""
    from ouroboros.tools.compact_context import standing_rows

    ctx = SimpleNamespace(active_context_mode="low")
    first = _fold(tmp_path, monkeypatch, ctx, [*_start(), *_read("one")], "fold-1", keep_unit_ids=[])
    request = [*first, _facts("round 3")]
    view = [*request, *_read("two"), *_read("three"), *_read("four")]
    tail = _fold(tmp_path, monkeypatch, ctx, view, "keep", keep_last_n=2)
    assert tail[:len(request)] == request and _notes(tail) == [3, len(request)]
    own = next(row for row in tail if (row.get("tool_calls") or [{}])[0].get("id") == "keep")
    assert json.loads(own["tool_calls"][0]["function"]["arguments"])["working_note"] == "Note written by keep."
    after = [*tail, _facts("round 5")]
    assert max(standing_rows(after)) == len(request) - 1  # nothing of the keep_last_n fold stands
    full = _fold(tmp_path, monkeypatch, ctx, [*after, *_read("five")], "fold-2", keep_unit_ids=[])
    assert full[:len(request) + 1] == after[:len(request) + 1] and _notes(full) == [3, len(request), len(request) + 1]
    assert not any(row.get("tool_call_id") in ("three", "four", "keep", "five") for row in full)


def test_fold_last_n_changes_only_the_tail_and_refuses_what_it_cannot_select(tmp_path, monkeypatch):
    """Test 4: fold_last_n folds the newest N tool units and everything after the first of them; every row before
    the first folded unit stays byte-identical, a superseded host row and a typed row that is no unit included (a
    fold changes nothing before its first replaced unit), and the tail fold leaves no anchor. Its argument errors
    change nothing."""
    from ouroboros.tools.compact_context import standing_rows

    ctx = SimpleNamespace(active_context_mode="low")
    typed = {**_facts("round 2"), "msg_id": "facts-2"}  # a host row the dialogue scope does not address
    view = [*_start(), *_read("one"), typed, *_read("two"), _facts("round 3"), *_read("three")]
    first_folded = len(view) - 2
    folded = _fold(tmp_path, monkeypatch, ctx, view, "tail", arrivals=[_facts("round 4")], fold_last_n=1)
    assert folded[:first_folded] == view[:first_folded] and _notes(folded) == [first_folded]
    assert not any(row.get("tool_call_id") == "three" for row in folded)
    assert sorted(standing_rows([*folded, _facts("round 5")])) == [2]  # raw units before it: no anchor
    two = _fold(tmp_path, monkeypatch, ctx, view, "tail-2", fold_last_n=2)
    assert two[:5] == view[:5] and _notes(two) == [6] and typed in two
    record_context_view(ctx, view, [])
    for selector, words in (({"fold_last_n": 0}, "at least 1"), ({"fold_last_n": True}, "at least 1"),
                            ({"fold_last_n": 1, "keep_last_n": 2}, "keep_last_n both select"),
                            ({"fold_last_n": 1, "keep_unit_ids": []}, "keep_unit_ids both select")):
        assert words in _compact_context(ctx, working_note="A note.", **selector)
    record_context_view(ctx, _start(), [])
    assert "no completed tool unit" in _compact_context(ctx, working_note="A note.", fold_last_n=1)
    assert ctx._pending_compaction is None  # no refused request was queued


def test_inspect_lists_standing_units_as_not_eligible_and_says_why(tmp_path, monkeypatch):
    """Test 5 (inspect): a unit of the start and a unit of a full fold's own rows are listed as not eligible with the
    fact of why; an ordinary unit stays eligible with no such fact."""
    ctx = SimpleNamespace(active_context_mode="low")
    first = _fold(tmp_path, monkeypatch, ctx, [*_start(), *_read("one")], "fold-1", keep_unit_ids=[])
    view = [*first, _facts("round 3"), *_read("two")]
    record_context_view(ctx, view, [])
    units = {unit["unit_id"]: unit for unit in json.loads(_compact_context(ctx, inspect=True))["units"]}
    note = _unit_at(view, 3).unit_id
    assert units[_unit_at(view, 2).unit_id]["stands"] == "start of the request"
    assert units[_unit_at(view, 4).unit_id]["stands"] == f"own rows of note {note}"
    assert not units[_unit_at(view, 2).unit_id]["eligible"] and not units[_unit_at(view, 4).unit_id]["eligible"]
    ordinary = units[_unit_at(view, len(view) - 2).unit_id]
    assert ordinary["eligible"] and "stands" not in ordinary


@pytest.mark.parametrize("route", ["anthropic", "responses"])
def test_the_own_calls_pointer_leaves_native_replay_and_keeps_a_complete_unit(tmp_path, monkeypatch, route):
    """Test 6: with a direct-Anthropic or a direct-Responses receipt on the fold's own call, the pointer edit makes
    the row project canonically (its receipt no longer validates) and the call with its result is one complete tool
    unit. Before the edit the same row validates (the guard's other direction)."""
    from ouroboros.anthropic_native_custody import _validated_receipt, retain_native_assistant_content
    from ouroboros.context_compaction import context_units
    from ouroboros.openai_responses_custody import retain_items, validated_items

    arguments = {"working_note": "Note written by native.", "keep_unit_ids": []}
    row = {"role": "assistant", "content": "", "tool_calls": [{"id": "native", "type": "function", "function": {
        "name": "compact_context", "arguments": json.dumps(arguments)}}]}
    if route == "anthropic":
        row = retain_native_assistant_content(row, [
            {"type": "thinking", "thinking": "private thought", "signature": "signature"},
            {"type": "tool_use", "id": "native", "name": "compact_context", "input": arguments}],
            {"provider": "anthropic", "base_url": "https://anthropic.example/v1", "resolved_model": "test-model"})
        valid = lambda message: _validated_receipt(message) is not None  # noqa: E731
    else:
        row = retain_items(row, [{"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque"},
                                 {"type": "function_call", "call_id": "native", "name": "compact_context",
                                  "arguments": json.dumps(arguments)}],
                           {"provider": "openai", "api_surface": "responses", "base_url": "https://api.openai.com/v1",
                            "resolved_model": "gpt-test"}, "completed")
        valid = lambda message: validated_items(message) is not None  # noqa: E731
    assert valid(row)
    ctx = SimpleNamespace(active_context_mode="low")
    folded = _fold(tmp_path, monkeypatch, ctx, [*_start(), *_read("one")], "native", own=row, keep_unit_ids=[])
    at = next(i for i, message in enumerate(folded) if (message.get("tool_calls") or [{}])[0].get("id") == "native")
    assert not valid(folded[at]) and "working_note" in folded[at]["tool_calls"][0]["function"]["arguments"]
    assert any(unit.start == at and unit.end == at + 1 for unit in context_units(folded, scope="dialogue"))


def test_an_untyped_start_row_takes_the_first_anchor_but_not_the_typed_start(tmp_path, monkeypatch):
    """A host notice written untyped at the start (as the capability omission notice is) is ordinary prose: the first
    full fold writes its note in its place, before the typed start rows, so that fold's own call does not follow the
    note and leaves no anchor (a disclosed gap). Both directions: the typed start rows still stand through the next
    full fold (a note is no answer that would end the start), and that fold's call points at its note and stands."""
    from ouroboros.tools.compact_context import NOTE_POINTER, standing_rows

    ctx = SimpleNamespace(active_context_mode="low")
    notice = {"role": "user", "content": "[SYSTEM NOTICE]\n[CAPABILITY_OMISSION_MANIFEST]\n- tools: missing credentials"}
    start = [*_source()[:2], notice, _facts("round 1")]
    first = _fold(tmp_path, monkeypatch, ctx, [*start, *_read("one")], "fold-1", keep_unit_ids=[])
    assert _notes(first) == [2] and first[3] == start[3]
    assert "Note written by fold-1." in first[4]["tool_calls"][0]["function"]["arguments"]  # no pointer
    request = [*first, _facts("round 3")]
    assert sorted(standing_rows(request)) == [3]
    second = _fold(tmp_path, monkeypatch, ctx, [*request, *_read("two")], "fold-2", keep_unit_ids=[])
    assert second[:4] == request[:4] and _notes(second) == [2, 4]
    assert NOTE_POINTER.split("{")[0] in second[5]["tool_calls"][0]["function"]["arguments"]
    assert sorted(standing_rows([*second, _facts("round 5")])) == [3, *range(5, len(second) + 1)]


def test_a_host_source_view_ends_the_standing_rows_and_the_next_full_fold_takes_what_follows_it(tmp_path, monkeypatch):
    """A restored source or an emergency record (a capsule with retention source_view) follows the keep selection, so
    a full fold removes it and writes its note there: it ends the standing rows like any removable unit. Both
    directions: the anchor before it stands, and the next full fold keeps that whole earlier request; the anchor after
    it (here a tail fold's own call, which points at its note) does not stand, and the next full fold takes its rows."""
    from ouroboros.context_compaction import _capsule_metadata
    from ouroboros.tools.compact_context import standing_rows
    from tests._working_notes import _note

    ctx = SimpleNamespace(active_context_mode="low")
    older = {**_note(tmp_path, "An earlier note.", "older"), "role": "assistant"}
    restored = copy.deepcopy(_note(tmp_path, "Restored exact source.", "restored"))
    meta = restored["content"][0]["_context_capsule"]
    meta.update(retention="source_view", authorship="host")
    meta.pop("written_at", None)
    assert _capsule_metadata(restored)[1]["retention"] == "source_view"
    first = _fold(tmp_path, monkeypatch, ctx, [*_start(), *_read("one")], "fold-1", keep_unit_ids=[])
    request = [*first, _facts("round 3")]
    view = [*request, older, restored, *_read("two")]
    tail = _fold(tmp_path, monkeypatch, ctx, view, "tail", fold_last_n=1)
    assert tail[:len(view) - 2] == view[:-2] and _notes(tail) == [3, len(request), len(request) + 1, len(request) + 2]
    after = [*tail, _facts("round 5")]
    assert sorted(standing_rows(after)) == [2, *range(4, len(request))]  # the tail's own rows after it do not stand
    full = _fold(tmp_path, monkeypatch, ctx, [*after, *_read("three")], "fold-2", keep_unit_ids=[])
    assert full[:len(request) + 1] == [*request, older] and restored not in full
    assert not any(row.get("tool_call_id") in ("tail", "three") for row in full)
