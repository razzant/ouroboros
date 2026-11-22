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

    from ouroboros import artifacts  # other direction: the receipt cannot be stored, so its row carries the lists
    store = artifacts.store_actor_source_bytes

    def refuse_the_receipt(*args, data=b"", **kwargs):
        if b'"candidate_materialization"' in data:
            raise OSError("disk full")
        return store(*args, data=data, **kwargs)

    monkeypatch.setattr(artifacts, "store_actor_source_bytes", refuse_the_receipt)
    record_context_view(ctx, messages, [])
    _compact_context(ctx, working_note="The read confirms the goal.", keep_unit_ids=[])
    inline, _ = _receipt(_run_round_compaction(copy.deepcopy(messages), _context(tmp_path, ctx, [], monkeypatch)))
    assert inline["receipt_source"] == "unavailable" and "at_source" not in inline
    assert inline["bodies_removed"] == stored["bodies_removed"] and len(inline["notes_kept"]) == 2
    assert set(inline["restore_checkpoint"]) == {"kind", "root", "path", "size", "sha256"}


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


# --- cheap folds: what stands through a fold (recognised by the note's text), the tail selector -------------------

def _long_read(call_id, text="Long exact evidence "):
    return [{"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function", "function": {
        "name": "read_file", "arguments": json.dumps({"path": f"{call_id}.py"})}}]},
        {"role": "tool", "tool_call_id": call_id, "content": text * 200}]


def _facts(text):
    from ouroboros.context_budget import bound_host_row
    from ouroboros.loop_messages import CONTEXT_FACTS_NAME

    return bound_host_row(CONTEXT_FACTS_NAME, "[CONTEXT_FACTS] " + text)


def _own_call(call_id, arguments):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function",
            "function": {"name": "compact_context", "arguments": json.dumps(arguments)}}]}


def _fold(tmp_path, monkeypatch, ctx, view, call_id, *, arrivals=(), own=None, root=None, **selector):
    """One fold the way Main applies it: the mind's call on the view it saw, its result, the boundary. The view's
    newer rows stand for rounds between two calls, so a roster note an earlier fold held back is no longer held."""
    ctx._roster_held = False
    record_context_view(ctx, view, [])
    arguments = {"working_note": f"Note written by {call_id}.", **selector}
    response = _compact_context(ctx, **arguments)
    assert "requested" in response, response
    current = [*copy.deepcopy(view), own or _own_call(call_id, arguments),
               {"role": "tool", "tool_call_id": call_id, "content": response}, *arrivals]
    folded = _run_round_compaction(current, _context(root or tmp_path / call_id, ctx, [], monkeypatch))
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


def _calls(rows):
    return {call.get("id") for row in rows for call in row.get("tool_calls") or ()}


def test_a_full_fold_stands_after_the_start_and_releasing_its_note_or_omitting_it_takes_its_rows(tmp_path, monkeypatch):
    """A full fold keeps the start's host rows and writes its note after them; its own call keeps the arguments the
    model sent, and the next full fold keeps that note's whole request (call, result, receipt, the next facts row)
    and writes after it. Both directions: replacing the note with keep_unit_ids [] takes its rows; replacing it with
    omitted selectors takes only its rows (a later fold's own rows and an ordinary read stay); an explicit
    keep_unit_ids list that omits a standing unit removes it."""
    from ouroboros.tools.compact_context import standing_rows

    ctx = SimpleNamespace(active_context_mode="low")
    start = _start()
    first = _fold(tmp_path, monkeypatch, ctx, [*start, *_long_read("one")], "fold-1", keep_unit_ids=[])
    at, = _notes(first)
    assert first[:at] == start and at == len(start)
    assert first[at]["content"][0]["text"].endswith("\nNote written by fold-1.")
    assert first[at + 1] == _own_call("fold-1", {"working_note": "Note written by fold-1.", "keep_unit_ids": []})
    request = [*first, _facts("round 3")]  # the request sent after that fold, up to the model's answer
    assert standing_rows(request) == {2: None, **dict.fromkeys(range(at + 1, len(request)), at)}

    second = _fold(tmp_path, monkeypatch, ctx, [*request, *_long_read("two")], "fold-2", keep_unit_ids=[])
    assert second[:len(request)] == request and _notes(second) == [at, len(request)]
    from ouroboros.tools.compact_context import receipt_row
    again = [_own_call("again", {"working_note": "Note written by fold-1.", "keep_unit_ids": []}),
             {"role": "tool", "tool_call_id": "again", "content": "Working view requested."},
             receipt_row({"status": "fit_rejected", "fit": {"accepted": False}})]  # a rejected call repeating the note
    assert not set(range(len(request), len(request) + 3)) & set(standing_rows([*request, *again]))
    taken = _fold(tmp_path, monkeypatch, ctx, [*request, *again, *_long_read("two")], "fold-7", keep_unit_ids=[])
    assert taken[:len(request)] == request and "again" not in _calls(taken)

    note_id = _unit_at(request, at).unit_id
    released = _fold(tmp_path, monkeypatch, ctx, [*request, *_long_read("two")], "fold-3", keep_unit_ids=[],
                     replace_note_ids=[note_id])
    assert released[:at] == start and _notes(released) == [at]  # its rows went with it
    assert "fold-1" not in _calls(released)
    later = [*second, _facts("round 5")]
    alone = _fold(tmp_path, monkeypatch, ctx, [*later, *_long_read("three")], "fold-5", replace_note_ids=[note_id])
    assert "fold-1" not in _calls(alone) and alone[:at] == start  # omitted selectors: only its own rows go
    assert {"fold-2", "three"} <= _calls(alone)  # the later note's call and an ordinary read stay
    assert [row for row in later[len(request):] if row in alone] == later[len(request):]  # fold-2's rows, its receipt
    omitted = _fold(tmp_path, monkeypatch, ctx, [*request, *_long_read("two")], "fold-4",
                    keep_unit_ids=[_unit_at(request, at + 1).unit_id])
    assert request[2] not in omitted and request[at + 1] in omitted  # the mind's explicit choice still decides
    interim = {"role": "assistant", "content": "An interim reply."}
    tail = _fold(tmp_path, monkeypatch, ctx, [*request, interim, *_long_read("two")], "fold-6", fold_last_n=2)
    assert "fold-1" in _calls(tail) and interim in tail and "two" not in _calls(tail)  # no standing unit counts
    from ouroboros.context_source_view import superseded_host_fact
    # The facts line counts a superseded row that does not stand (round 4 after the read, which the row being written
    # supersedes) and not the standing rounds 1 and 3 before it.
    assert superseded_host_fact([*request, *_long_read("two"), _facts("round 4")]).startswith("superseded host rows 1 ")



def test_two_notes_with_equal_text_each_own_only_their_folds_rows(tmp_path, monkeypatch):
    """Ownership follows the note the receipt row names, not the note's text: after a replacement writes a note with
    the same text as a later note, at an earlier position, replacing either note releases only its own fold's rows."""
    ctx = SimpleNamespace(active_context_mode="low")
    first = _fold(tmp_path, monkeypatch, ctx, [*_start(), *_long_read("one")], "fold-1", keep_unit_ids=[])
    n1 = _unit_at(first, 3).unit_id
    second = _fold(tmp_path, monkeypatch, ctx, [*first, _facts("round 3"), *_long_read("two")], "fold-2",
                   keep_unit_ids=[])
    third = _fold(tmp_path, monkeypatch, ctx, [*second, _facts("round 5")], "fold-3", replace_note_ids=[n1],
                  working_note="Note written by fold-2.")
    n3, n2 = _notes(third)[:2]
    assert n3 < n2 and "fold-1" not in _calls(third) and {"fold-2", "fold-3"} <= _calls(third)
    view = [*third, _facts("round 7"), *_long_read("four")]
    for replaced, gone, kept in ((n3, "fold-3", "fold-2"), (n2, "fold-2", "fold-3")):
        out = _fold(tmp_path, monkeypatch, ctx, view, f"merge-{gone}", replace_note_ids=[_unit_at(view, replaced).unit_id])
        assert gone not in _calls(out) and kept in _calls(out) and "four" in _calls(out), (gone, kept)

def test_the_owners_words_inside_a_folded_stretch_leave_the_fold_standing_and_a_shared_call_row_does_not(
        tmp_path, monkeypatch):
    """A full fold over ``read A, [Message from my human], read B`` writes its note at read A and keeps the owner's
    words between the note and its own call: that fold's rows stand and the next full fold keeps its whole request,
    the owner row byte-identical throughout. Other direction: a fold whose call shares its assistant row with another
    call leaves nothing that stands, and the next full fold takes its rows."""
    from ouroboros.tools.compact_context import standing_rows
    from tests._context_reclaim_fixtures import _owner

    ctx = SimpleNamespace(active_context_mode="low")
    owner = _owner("[Message from my human]\nKeep the migration plan as it is.")
    view = [*_start(), *_long_read("a"), owner, *_long_read("b")]
    first = _fold(tmp_path, monkeypatch, ctx, view, "fold-1", keep_unit_ids=[])
    assert _notes(first) == [3] and first[4] == owner and "fold-1" in _calls([first[5]])
    request = [*first, _facts("round 3")]
    assert sorted(standing_rows(request)) == [2, 4, *range(5, len(request))]  # the owner row lies before the first answer
    second = _fold(tmp_path, monkeypatch, ctx, [*request, *_long_read("c")], "fold-2", keep_unit_ids=[])
    assert second[:len(request)] == request and second[4] == owner

    arguments = {"working_note": "Note written by shared.", "keep_unit_ids": []}
    shared = _own_call("shared", arguments)
    shared["tool_calls"].append({"id": "beside", "type": "function", "function": {
        "name": "read_file", "arguments": json.dumps({"path": "beside.py"})}})
    folded = _fold(tmp_path, monkeypatch, ctx, view, "shared", own=shared, keep_unit_ids=[],
                   arrivals=[{"role": "tool", "tool_call_id": "beside", "content": "Another result."}])
    after = [*folded, _facts("round 3")]
    assert sorted(standing_rows(after)) == [2, 4]  # the start and the owner row only
    full = _fold(tmp_path, monkeypatch, ctx, [*after, *_long_read("c")], "fold-3", keep_unit_ids=[])
    assert not {"shared", "beside"} & _calls(full) and owner in full


def test_keep_last_n_leaves_no_anchor_and_the_next_full_fold_takes_its_tail(tmp_path, monkeypatch):
    """A keep_last_n fold keeps an earlier full fold's standing rows untouched, writes its note after them and leaves
    no rows that stand (raw units sit between its note and its call); the next full fold folds its kept tail and its
    own rows, and writes its note right after the keep_last_n fold's note."""
    from ouroboros.tools.compact_context import standing_rows

    ctx = SimpleNamespace(active_context_mode="low")
    first = _fold(tmp_path, monkeypatch, ctx, [*_start(), *_long_read("one")], "fold-1", keep_unit_ids=[])
    request = [*first, _facts("round 3")]
    view = [*request, *_long_read("two"), *_long_read("three"), *_long_read("four")]
    tail = _fold(tmp_path, monkeypatch, ctx, view, "keep", keep_last_n=2)
    assert tail[:len(request)] == request and _notes(tail) == [3, len(request)]
    after = [*tail, _facts("round 5")]
    assert max(standing_rows(after)) == len(request) - 1  # nothing of the keep_last_n fold stands
    full = _fold(tmp_path, monkeypatch, ctx, [*after, *_long_read("five")], "fold-2", keep_unit_ids=[])
    assert full[:len(request) + 1] == after[:len(request) + 1] and _notes(full) == [3, len(request), len(request) + 1]
    assert not {"three", "four", "keep", "five"} & _calls(full)


def test_fold_last_n_changes_only_the_tail_and_refuses_what_it_cannot_select(tmp_path, monkeypatch):
    """fold_last_n folds the newest N tool units and everything after the first of them; every row before the first
    folded unit stays byte-identical, a superseded host row and a typed row that is no unit included (a fold changes
    nothing before its first replaced unit), and raw units before its note leave no rows that stand. It needs the
    mind's note: a missing or blank working_note is an argument error, with view_address too, while a written note
    applies. Its argument errors queue nothing."""
    from ouroboros.tools.compact_context import standing_rows

    ctx = SimpleNamespace(active_context_mode="low")
    typed = {**_facts("round 2"), "msg_id": "facts-2"}  # a host row the dialogue scope does not address
    view = [*_start(), *_long_read("one"), typed, *_long_read("two"), _facts("round 3"), *_long_read("three")]
    first_folded = len(view) - 2
    folded = _fold(tmp_path, monkeypatch, ctx, view, "tail", arrivals=[_facts("round 4")], fold_last_n=1)
    assert folded[:first_folded] == view[:first_folded] and _notes(folded) == [first_folded]
    assert "three" not in _calls(folded)
    assert sorted(standing_rows([*folded, _facts("round 5")])) == [2]  # raw units before it
    two = _fold(tmp_path, monkeypatch, ctx, view, "tail-2", fold_last_n=2)
    assert two[:5] == view[:5] and _notes(two) == [6] and typed in two
    record_context_view(ctx, view, [])
    for selector, words in (({"fold_last_n": 0}, "at least 1"), ({"fold_last_n": True}, "at least 1"),
                            ({"fold_last_n": 1, "keep_last_n": 2}, "keep_last_n both select"),
                            ({"fold_last_n": 1, "keep_unit_ids": []}, "keep_unit_ids both select")):
        assert words in _compact_context(ctx, working_note="A note.", **selector)
    for arguments in ({"working_note": "   ", "fold_last_n": 1}, {"working_note": "", "fold_last_n": 1},
                      {"fold_last_n": 1}, {"view_address": ["F5:page-000"], "fold_last_n": 1}):
        assert _compact_context(ctx, **arguments).startswith("authored_note_required"), arguments
    assert ctx._pending_compaction is None
    record_context_view(ctx, _start(), [])
    assert "no completed tool unit" in _compact_context(ctx, working_note="A note.", fold_last_n=1)
    assert ctx._pending_compaction is None  # no refused request was queued


def test_inspect_lists_standing_units_as_not_eligible_and_says_why(tmp_path, monkeypatch):
    """Inspect: a unit of the start and a unit of a full fold's own rows are listed as not eligible with the fact of
    why; an ordinary unit stays eligible with no such fact."""
    ctx = SimpleNamespace(active_context_mode="low")
    first = _fold(tmp_path, monkeypatch, ctx, [*_start(), *_long_read("one")], "fold-1", keep_unit_ids=[])
    view = [*first, _facts("round 3"), *_long_read("two")]
    record_context_view(ctx, view, [])
    units = {unit["unit_id"]: unit for unit in json.loads(_compact_context(ctx, inspect=True))["units"]}
    note = _unit_at(view, 3).unit_id
    assert units[_unit_at(view, 2).unit_id]["stands"] == "start of the request"
    assert units[_unit_at(view, 4).unit_id]["stands"] == f"own rows of note {note}"
    assert not units[_unit_at(view, 2).unit_id]["eligible"] and not units[_unit_at(view, 4).unit_id]["eligible"]
    ordinary = units[_unit_at(view, len(view) - 2).unit_id]
    assert ordinary["eligible"] and "stands" not in ordinary


@pytest.mark.parametrize("route", ["anthropic", "responses", "claudexor"])
def test_a_fold_keeps_its_own_call_as_sent_with_its_native_envelope_and_its_rows_stand(tmp_path, monkeypatch, route):
    """A fold rewrites no call the model sent. With a direct-Anthropic receipt, a direct-Responses receipt or a
    Claudexor ``nativeContinuation`` envelope on the fold's own call, the call row after the fold equals the row
    before it key for key, the direct receipts still validate, and the call with its result is one complete tool
    unit. Other direction: that fold's rows stand, and the next full fold keeps the whole earlier request."""
    from ouroboros.anthropic_native_custody import (_validated_receipt, anthropic_tool_unit_active,
                                                    retain_native_assistant_content)
    from ouroboros.context_compaction import context_units
    from ouroboros.openai_responses_custody import responses_tool_unit_active, retain_items, validated_items
    from ouroboros.tools.compact_context import standing_rows

    arguments = {"working_note": "Note written by native.", "keep_unit_ids": []}
    row, valid = _own_call("native", arguments), lambda message: True  # noqa: E731
    if route == "anthropic":
        row = retain_native_assistant_content(row, [
            {"type": "thinking", "thinking": "private thought", "signature": "signature"},
            {"type": "tool_use", "id": "native", "name": "compact_context", "input": arguments}],
            {"provider": "anthropic", "base_url": "https://anthropic.example/v1", "resolved_model": "test-model"})
        valid = lambda message: _validated_receipt(message) is not None  # noqa: E731
    elif route == "responses":
        row = retain_items(row, [{"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque"},
                                 {"type": "function_call", "call_id": "native", "name": "compact_context",
                                  "arguments": json.dumps(arguments)}],
                           {"provider": "openai", "api_surface": "responses", "base_url": "https://api.openai.com/v1",
                            "resolved_model": "gpt-test"}, "completed")
        valid = lambda message: validated_items(message) is not None  # noqa: E731
    else:  # the envelope Claudexor returns with a Codex turn (tests/fixtures/llm_wire/claudexor_capability)
        row["nativeContinuation"] = {"route": {"source": "codex", "credentialProfileId": "a", "accountFingerprint": "f",
                                               "model": "gpt-test"}, "format": "codex.responses.v1", "payload": [
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque"},
            {"type": "function_call", "call_id": "native", "name": "compact_context",
             "arguments": json.dumps(arguments)}]}
    assert valid(row)
    sent = copy.deepcopy(row)
    ctx = SimpleNamespace(active_context_mode="low")
    folded = _fold(tmp_path, monkeypatch, ctx, [*_start(), *_long_read("one")], "native", own=row, keep_unit_ids=[])
    at = next(i for i, message in enumerate(folded) if "native" in _calls([message]))
    assert folded[at] == sent and valid(folded[at]) and folded[at + 1]["tool_call_id"] == "native"
    # One complete tool unit; a direct receipt keeps it an active native replay until a later answer consumes it.
    assert any(unit.start == at and unit.end == at + 1 for unit in context_units(folded, scope="dialogue")) or (
        anthropic_tool_unit_active if route == "anthropic" else responses_tool_unit_active)(folded, at, at + 1)
    request = [*folded, _facts("round 3")]
    assert set(range(at, len(request))) <= set(standing_rows(request))
    second = _fold(tmp_path, monkeypatch, ctx, [*request, *_long_read("two")], "fold-2", keep_unit_ids=[])
    assert second[:len(request)] == request


def test_the_start_stands_typed_or_not_and_its_handed_over_accounts_stay_foldable(tmp_path, monkeypatch):
    """Every row between the assignment and the first answer stands, typed or not, except the accounts handed over
    for the mind to fold (review-history sources, a predecessor's notes, a late answer's asker notes). Both
    directions: an untyped host notice stands while keep_unit_ids [] takes a review body, a predecessor_notes row and
    a late_answer_asker_notes row, the other start rows byte-identical, and that fold's rows stand; an explicit
    keep_unit_ids list that omits the notice takes it. Where the body comes first (as capture places it), the start
    rows that survive between the note and its call do not end the walk: the first fold's rows stand already."""
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
    from ouroboros.main_context_authority import LATE_ANSWER_NOTES_NAME, PREDECESSOR_NOTES_NAME
    from ouroboros.review_history_view import _body_message
    from ouroboros.tools.compact_context import standing_rows

    ctx = SimpleNamespace(active_context_mode="low", drive_root=tmp_path, task_id="view-task")  # a fold over a review body
    notice = {"role": "user", "content": "[SYSTEM NOTICE]\n[CAPABILITY_OMISSION_MANIFEST]\n- tools: missing credentials"}
    body = _body_message({"binding": {"kind": "review_history_body", "field": "findings", "content_sha256": "b" * 64,
                                      "source_ref": {"path": "reviews/wave-1.json", "size": 10, "sha256": "a" * 64}},
                          "value": "Reviewer finding: the plan skips the migration."}, "view-task")
    inherited = [{"role": "user", HOST_CONTEXT_KIND_KEY: kind, "content": text} for kind, text in (
        (PREDECESSOR_NOTES_NAME, "[Predecessor notes]\nThe earlier work chose the slow path."),
        (LATE_ANSWER_NOTES_NAME, "[Late answer: the asker's notes]\nThe asker wanted the numbers."))]
    start = [*_source()[:2], notice, _facts("round 1"), body, *inherited]
    assert sorted(standing_rows([*start, *_long_read("one")])) == [2, 3]
    first = _fold(tmp_path, monkeypatch, ctx, [*start, *_long_read("one")], "fold-1", keep_unit_ids=[])
    assert first[:4] == start[:4] and _notes(first) == [4] and not [row for row in (body, *inherited) if row in first]
    request = [*first, _facts("round 3")]
    assert sorted(standing_rows(request)) == [2, 3, *range(5, len(request))]
    second = _fold(tmp_path, monkeypatch, ctx, [*request, *_long_read("two")], "fold-2", keep_unit_ids=[])
    assert second[:len(request)] == request and _notes(second) == [4, len(request)]
    omitted = _fold(tmp_path, monkeypatch, ctx, [*start, *_long_read("one")], "fold-3",
                    keep_unit_ids=[_unit_at(start, 3).unit_id])
    assert notice not in omitted and start[3] in omitted

    captured = [*_source()[:2], body, notice, _facts("round 1")]  # capture writes the bodies right after the assignment
    early = _fold(tmp_path, monkeypatch, ctx, [*captured, *_long_read("one")], "early-1", keep_unit_ids=[])
    assert _notes(early) == [2] and early[3:5] == captured[3:5]  # the note at the body, before the other start rows
    after = [*early, _facts("round 3")]
    assert sorted(standing_rows(after)) == [3, 4, *range(5, len(after))]
    later = _fold(tmp_path, monkeypatch, ctx, [*after, *_long_read("two")], "early-2", keep_unit_ids=[])
    assert later[:len(after)] == after and _notes(later) == [2, len(after)]


def test_a_host_source_view_ends_the_standing_rows_and_the_next_full_fold_takes_what_follows_it(tmp_path, monkeypatch):
    """A restored source or an emergency record (a capsule with retention source_view) follows the keep selection, so
    a full fold removes it and writes its note there: it ends the standing rows like any removable unit. Both
    directions: the rows that stand before it stand, and the next full fold keeps that whole earlier request; a fold
    after it does not stand, and the next full fold takes its rows."""
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
    first = _fold(tmp_path, monkeypatch, ctx, [*_start(), *_long_read("one")], "fold-1", keep_unit_ids=[])
    request = [*first, _facts("round 3")]
    view = [*request, older, restored, *_long_read("two")]
    tail = _fold(tmp_path, monkeypatch, ctx, view, "tail", fold_last_n=1)
    assert tail[:len(view) - 2] == view[:-2] and _notes(tail) == [3, len(request), len(request) + 1, len(request) + 2]
    after = [*tail, _facts("round 5")]
    assert sorted(standing_rows(after)) == [2, *range(4, len(request))]  # the tail's own rows after it do not stand
    full = _fold(tmp_path, monkeypatch, ctx, [*after, *_long_read("three")], "fold-2", keep_unit_ids=[])
    assert full[:len(request) + 1] == [*request, older] and restored not in full
    assert not {"tail", "three"} & _calls(full)


def test_a_source_view_a_fold_restores_after_earlier_rows_that_stand_stays_foldable(tmp_path, monkeypatch):
    """A fold with restore_unit_refs writes the restored source view before its note, after an earlier fold's rows:
    the source view does not stand (inspect lists it eligible), and neither do that fold's own rows; the next full
    fold removes the source view and those rows. Other direction: the earlier fold's rows before it stand and the
    next full fold keeps them."""
    from ouroboros.context_compaction import _capsule_metadata
    from ouroboros.tools.compact_context import standing_rows

    ctx = SimpleNamespace(active_context_mode="low")
    first = _fold(tmp_path, monkeypatch, ctx, [*_start(), *_long_read("one")], "fold-1", keep_unit_ids=[],
                  root=tmp_path)
    stored = ctx._context_view_receipt
    ref = {"checkpoint_ref": stored["restore_checkpoint"], **stored["bodies_removed"][0]["address"]}
    request = [*first, _facts("round 3")]
    second = _fold(tmp_path, monkeypatch, ctx, [*request, *_long_read("two")], "fold-2", keep_unit_ids=[],
                   restore_unit_refs=[ref], root=tmp_path)
    source = len(request)
    assert second[:source] == request and _capsule_metadata(second[source])[1]["retention"] == "source_view"
    view = [*second, _facts("round 5")]
    standing = standing_rows(view)
    assert set(range(4, source)) <= set(standing) and not set(range(source, len(view))) & set(standing)
    record_context_view(ctx, [*view, *_long_read("three")], [])
    units = {unit["unit_id"]: unit for unit in json.loads(_compact_context(ctx, inspect=True))["units"]}
    assert units[_unit_at(view, source).unit_id]["eligible"] and "stands" not in units[_unit_at(view, source).unit_id]
    third = _fold(tmp_path, monkeypatch, ctx, [*view, *_long_read("three")], "fold-3", keep_unit_ids=[], root=tmp_path)
    assert third[:source] == request and second[source] not in third and "fold-2" not in _calls(third)
    # A handed-over account arriving in a fold's own round (a late answer's asker notes) leaves that fold nothing
    # that stands, and the next full fold takes the account and the fold's rows.
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
    from ouroboros.main_context_authority import LATE_ANSWER_NOTES_NAME
    late = {"role": "user", HOST_CONTEXT_KIND_KEY: LATE_ANSWER_NOTES_NAME, "content": "[Late answer]\nThe asker's notes."}
    arrived = [*first, late, _facts("round 3")]
    assert sorted(standing_rows(arrived)) == [2]
    folded = _fold(tmp_path, monkeypatch, ctx, [*arrived, *_long_read("two")], "fold-4", keep_unit_ids=[], root=tmp_path)
    assert folded[:4] == first[:4] and late not in folded and "fold-1" not in _calls(folded)


def test_a_folds_leftover_rows_after_an_emergency_record_took_its_call_are_not_start_rows(tmp_path, monkeypatch):
    """When an emergency record (a host source view) replaced a fold's own call and result, the receipt and facts
    rows left after it are not start rows: replacing the note takes them. Other direction: the start's own rows
    before the note stay."""
    from ouroboros.tools.compact_context import standing_rows
    from tests._working_notes import _note

    ctx = SimpleNamespace(active_context_mode="low")
    first = _fold(tmp_path, monkeypatch, ctx, [*_start(), *_long_read("one")], "fold-1", keep_unit_ids=[])
    at, = _notes(first)
    record = copy.deepcopy(_note(tmp_path, "Exact source of the fold's call.", "emergency"))
    record["content"][0]["_context_capsule"].update(retention="source_view", authorship="host")
    view = [*first[:at + 1], record, *first[at + 3:], _facts("round 3")]  # its call and result became the record
    leftover = [row for row in view[at + 2:] if str(row.get("content")).startswith(("[Context view receipt]",
                                                                                    "[CONTEXT_FACTS]"))]
    assert len(leftover) == 2
    assert sorted(standing_rows(view)) == [2]
    folded = _fold(tmp_path, monkeypatch, ctx, [*view, *_long_read("two")], "fold-2", keep_unit_ids=[],
                   replace_note_ids=[_unit_at(view, at).unit_id])
    assert folded[:at] == view[:at] and not [row for row in leftover if row in folded]


def test_a_reviewer_style_notice_after_an_own_call_stands_nothing_of_that_fold(tmp_path):
    """A fold's own call followed by a notice that is not the host's receipt row (a reviewer session's fold), or by a
    host receipt row that names no note this request wrote (a transfer that reused a note, a fold made before
    receipts named their note), leaves nothing of that fold standing; the same rows with the host's receipt row
    naming the note stand (the other direction). The writer names a note only when the request wrote it."""
    from ouroboros.context_compaction import _capsule_metadata
    from ouroboros.tools.compact_context import receipt_head, receipt_row, standing_rows
    from tests._working_notes import _note

    note = {**_note(tmp_path, "Note written by reviewer."), "role": "assistant"}
    rows = [*_start(), note, _own_call("reviewer", {"working_note": "Note written by reviewer."}),
            {"role": "tool", "tool_call_id": "reviewer", "content": "Working view requested."}]
    reviewer = [*rows, {"role": "user", "content": "[Review context view]\n{}"}, _facts("round 3")]
    note_id = _capsule_metadata(note)[1]["unit_id"]
    host = [*rows, receipt_row({"status": "applied", "note": note_id}), _facts("round 3")]
    reused = [*rows, receipt_row({"status": "applied"}), _facts("round 3")]
    assert sorted(standing_rows(reviewer)) == sorted(standing_rows(reused)) == [2]
    assert sorted(standing_rows(host)) == [2, *range(4, len(host))]
    receipt = {"status": "applied", "selection_fingerprint": note_id.removeprefix("view:")}
    assert receipt_head(receipt, _start(), rows) == {"status": "applied", "note": note_id}  # this request wrote it
    assert receipt_head(receipt, rows, rows) == {"status": "applied"}  # a transfer that reused the observed note


from tests.test_review_view_integration import harness as _review_harness  # noqa: E402 -- the recorded plan-review fixture

harness = _review_harness


def test_no_fold_changes_a_review_index_before_its_note_and_a_stale_one_after_it_leaves(harness, tmp_path, monkeypatch):
    """At the fold's seam, the review-index refresh obeys the positional rule in both directions: a stale start index,
    a full fold, a newer index appended by a later review, a second full fold and a tail fold (fold_last_n=1) leave
    every row before each fold's note byte-identical; a stale index copy after the tail fold's note leaves."""
    from ouroboros import review_history_view as rv
    from tests.test_review_view_integration import _capture, _three

    ctx, _ = _three(harness)
    ctx.active_context_mode = "low"
    now = next(row for row in _capture(ctx) if rv.REVIEW_CONTEXT_INDEX_KEY in row)
    old = rv._index_message({"decision_rows": []}, "plan", task_id=ctx.task_id)
    assert old != now
    view = [*_source()[:2], old, _facts("round 1"), *_long_read("r0")]
    first = _fold(tmp_path, monkeypatch, ctx, view, "fold-1", keep_unit_ids=[])
    at = _notes(first)[0]
    assert first[:at] == view[:at] and old in first
    view = [*first, _facts("round 3"), *_long_read("r1"), now, *_long_read("r2")]
    second = _fold(tmp_path, monkeypatch, ctx, view, "fold-2", keep_unit_ids=[])
    at = _notes(second)[-1]
    assert second[:at] == view[:at] and old in second
    late = {**copy.deepcopy(old), "content": old["content"]}
    view = [*second, _facts("round 5"), *_long_read("r3"), *_long_read("r4"), late]
    tail = _fold(tmp_path, monkeypatch, ctx, view, "tail", fold_last_n=1)
    at = _notes(tail)[-1]
    assert tail[:at] == view[:at] and at == len(view) - 3
    assert sum(row == old for row in tail) == 1 and tail[2] == old  # the copy after the note left, the start's stays
