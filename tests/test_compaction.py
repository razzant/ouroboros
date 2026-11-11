"""Focused contract tests for the unit reader and the mind's authored fold."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from ouroboros import context_compaction as cc
from ouroboros.context_budget import ContextReclaimRequest


def _unit(call_id: str, *, argument: str = "a", result: str = "r") -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": f"reasoning-{call_id}",
            "tool_calls": [{
                "id": call_id,
                "function": {"name": "read_file", "arguments": argument},
            }],
        },
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


def _large_unit(call_id: str, marker: str = "x") -> list[dict]:
    return _unit(
        call_id,
        argument=json.dumps({"content": marker * 4_000 + f"-arg-tail-{call_id}"}),
        result=marker * 4_000 + f"-result-tail-{call_id}",
    )


def _stamp_trace(messages, ref):
    from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY, make_tool_result_record
    row = messages[-1]
    row[TOOL_RESULT_RECORD_KEY] = make_tool_result_record(
        {"tool_call_id": row["tool_call_id"], "invocation_id": "fixture-invocation", "trace_ref": ref}, row["content"])


def _request(messages: list[dict], *, goal: int = 1_000_000) -> ContextReclaimRequest:
    return ContextReclaimRequest(
        route_fp="main-route",
        round_id="round-7",
        transcript_sha256=cc.context_reclaim_transcript_sha256(messages),
        measurement_basis="fresh_route_usage",
        measurement_density=1.0,
        reclaim_goal_tokens=goal,
        allow_partial_shrink=True,
    )


def _authored(messages: list[dict], *, note: str = "I read the evidence; the conclusion holds.", keep=()) -> ContextReclaimRequest:
    """The mind's own fold of the observed view: a note, the units it keeps raw."""
    return replace(_request(messages), working_note=note,
                   expected_view_revision=cc.context_reclaim_transcript_sha256(messages), keep_unit_ids=keep)


def _fold(messages: list[dict], request: ContextReclaimRequest, tmp_path):
    return cc.compact_tool_history_llm(
        messages, request=request, observed_messages=messages, tool_schemas=[],
        fit_candidate=lambda *_: {"accepted": True}, drive_root=tmp_path, task_id="compaction")


def test_atomic_units_require_all_and_only_contiguous_matching_results():
    complete = _unit("ok", argument="{}", result="done")
    missing = _unit("missing", argument="{}", result="never")[:1]
    delayed = _unit("late", argument="{}", result="late-result")
    duplicate = [{
        "role": "assistant",
        "tool_calls": [
            {"id": "dup", "function": {"name": "x", "arguments": "{}"}},
            {"id": "dup", "function": {"name": "y", "arguments": "{}"}},
        ],
    }, {"role": "tool", "tool_call_id": "dup", "content": "one"}]
    messages = [
        *complete,
        {"role": "user", "content": "hard boundary"},
        *missing,
        {"role": "user", "content": "interrupt"},
        delayed[1],
        *duplicate,
        *_unit("final", argument="{}", result="done"),
    ]

    units = cc._atomic_units(messages)

    assert [(unit.start, unit.end) for unit in units] == [(0, 1), (8, 9)]
    assert all(messages[unit.start]["role"] == "assistant" for unit in units)
    assert messages[2] == {"role": "user", "content": "hard boundary"}


@pytest.mark.parametrize(
    "malformed_call",
    ["not-a-tool-call", {"id": "bad", "function": "not-a-function"}],
)
def test_malformed_tool_call_set_stays_raw_without_crashing(malformed_call):
    messages = [
        {"role": "assistant", "tool_calls": [malformed_call]},
        {"role": "tool", "tool_call_id": "bad", "content": "result"},
    ]

    assert cc._atomic_units(messages) == ()


def test_atomic_source_keeps_complete_arguments_results_and_trace_ref():
    messages = _large_unit("call-complete", "z")
    trace_ref = {"path": "calls/tool.json", "sha256": "d" * 64}
    _stamp_trace(messages, trace_ref)

    unit = cc._atomic_units(
        messages,
        trace_refs_by_tool_call_id={"call-complete": trace_ref},
    )[0]

    assert "-arg-tail-call-complete" in unit.source_text
    assert "-result-tail-call-complete" in unit.source_text
    assert "CONTENT_OMITTED" not in unit.source_text
    assert "LONG_STRING" not in unit.source_text
    assert unit.source_refs == (trace_ref,)


def test_a_request_without_a_note_is_refused_before_any_checkpoint(monkeypatch, tmp_path):
    """Both directions: no helper writes the note, so a note-less request changes nothing and
    writes no checkpoint; the same selection with the mind's note is materialized."""
    messages = _large_unit("note-less")
    before = copy.deepcopy(messages)
    monkeypatch.setattr(cc, "_persist_reclaim_checkpoint",
                        lambda *_a, **_k: pytest.fail("a refused request must not checkpoint"))

    rebuilt, receipt = _fold(messages, _request(messages), tmp_path)

    assert rebuilt is messages and messages == before
    assert receipt.status == "authored_note_required"
    assert receipt.before_transcript_sha256 == receipt.after_transcript_sha256
    monkeypatch.setattr(cc, "_persist_reclaim_checkpoint",
                        lambda *_a, **_k: {"path": "calls/checkpoint.json", "sha256": "c" * 64})
    folded, applied = _fold(messages, _authored(messages), tmp_path)
    assert applied.status == "applied" and len(folded) == 1 and messages == before


def test_an_authored_fold_checkpoints_the_exact_source_before_its_capsule(monkeypatch, tmp_path):
    events: list[str] = []
    messages = _large_unit("ordered")
    _stamp_trace(messages, {"path": "calls/ordered.json", "sha256": "d" * 64})

    def checkpoint(*_args, **_kwargs):
        events.append("checkpoint")
        return {"path": "calls/checkpoint.json", "sha256": "c" * 64}

    monkeypatch.setattr(cc, "_persist_reclaim_checkpoint", checkpoint)
    rebuilt, receipt = _fold(messages, _authored(messages), tmp_path)

    assert events == ["checkpoint"]
    assert receipt.status == "applied"
    assert len(rebuilt) == 1 and rebuilt[0]["role"] == "assistant"
    capsule = rebuilt[0]["content"][0]["_context_capsule"]
    assert rebuilt[0]["content"][0]["text"].startswith("[Actor-authored working view; exact source retained by checkpoint]\n")
    assert capsule["authorship"] == "actor" and capsule["generation"] == 1
    assert capsule["retention"] == "summarized"
    assert capsule["checkpoint_ref"]["sha256"] == "c" * 64
    assert {"path": "calls/ordered.json", "sha256": "d" * 64} in capsule["source_refs"]
    assert capsule["parts"][0]["start_char"] == 0
    assert capsule["parts"][-1]["end_char"] == capsule["source_length_chars"]
    assert receipt.capsule_refs[0]["generation"] == 1


def test_checkpoint_failure_keeps_the_source_raw(monkeypatch, tmp_path):
    messages = _large_unit("checkpoint-fail")
    monkeypatch.setattr(cc, "_persist_reclaim_checkpoint", lambda *_a, **_k: None)

    rebuilt, receipt = _fold(messages, _authored(messages), tmp_path)

    assert rebuilt is messages
    assert receipt.status == "checkpoint_failed"


def test_a_second_fold_over_an_actor_note_keeps_its_lineage_and_a_corrupt_capsule_stays_raw(tmp_path):
    messages = _large_unit("generation", "g")
    first, first_receipt = _fold(messages, _authored(messages, note="First account. " * 200), tmp_path)
    named = replace(_authored(first, note="Merged account."), replace_note_ids=(cc.context_units(first)[0].unit_id,))
    second, second_receipt = _fold(first, named, tmp_path)

    first_meta = first[0]["content"][0]["_context_capsule"]
    second_meta = second[0]["content"][0]["_context_capsule"]
    assert first_receipt.status == second_receipt.status == "applied"
    assert second_meta["generation"] == 2
    assert set(first_meta["source_hashes"]) < set(second_meta["source_hashes"])
    assert all(ref in second_meta["source_refs"] for ref in first_meta["source_refs"])

    corrupt = copy.deepcopy(second)
    corrupt[0]["content"][0]["_context_capsule"]["source_length_chars"] += 1
    assert cc._atomic_units(corrupt) == ()

    corrupt_hash = copy.deepcopy(second)
    corrupt_hash[0]["content"][0]["_context_capsule"]["source_hashes"][0] = "broken"
    assert cc._atomic_units(corrupt_hash) == ()


def test_count_only_compact_context_answers_authored_note_required_and_advertises_no_helper():
    """Both directions: a count without a note queues nothing and names what is missing; the
    same count beside the mind's note queues the authored view."""
    from ouroboros.tools.compact_context import _compact_context, get_tools, record_context_view

    ctx = SimpleNamespace(active_context_mode="max", _active_builtin_tool_result=None)
    record_context_view(ctx, _unit("seen"), [])
    result = _compact_context(ctx, keep_last_n=4)
    description = get_tools()[0].schema["description"]

    assert getattr(ctx, "_pending_compaction", None) is None
    assert "authored_note_required" in result and "working_note" in result
    assert ctx._active_builtin_tool_result.status == "error"
    assert ctx._active_builtin_tool_result.code == "TOOL_ARG_ERROR"
    assert "no helper summarizes for you" in description.lower()
    assert "summaries without a note" not in description.lower()
    assert "raw evidence remains retrievable" in description.lower()
    assert "no information is lost" not in description.lower()
    assert "requested" in _compact_context(ctx, keep_last_n=4, working_note="Four units stay raw.")
    assert isinstance(ctx._pending_compaction, dict) and ctx._pending_compaction["working_note"] == "Four units stay raw."


def _note(tmp_path, text: str, call_id: str = "noted") -> dict:
    """One earlier note of the mind: the capsule an applied fold left in place of its source."""
    source = _large_unit(call_id)
    folded, receipt = _fold(source, _authored(source, note=text), tmp_path)
    assert receipt.status == "applied" and len(folded) == 1
    return folded[0]


def _unattributed(capsule: dict) -> dict:
    """The same capsule as a release before authorship metadata wrote it: still a valid note, attributed to no one."""
    legacy = copy.deepcopy(capsule)
    meta = legacy["content"][0]["_context_capsule"]
    del meta["authorship"]
    meta["summary_contract_digest"] = cc._LEGACY_CAPSULE_CONTRACT_DIGEST
    assert cc._capsule_metadata(legacy)[1] is not None
    return legacy


def test_an_empty_keep_selection_folds_tool_units_but_keeps_every_earlier_note(tmp_path):
    """Both directions: keep_unit_ids=[] still folds every eligible tool unit, and no earlier note leaves with
    them, whatever its authorship; a capsule written before authorship existed is not silently merged either."""
    goal = _note(tmp_path, "Goal: a few elegant generalizable prompt fixes, no new framework.", "goal")
    older = _unattributed(_note(tmp_path, "An earlier release kept this finding.", "older"))
    view = [goal, older, *_large_unit("read", "r")]
    units, before = cc.context_units(view, scope="dialogue"), copy.deepcopy(view)

    folded, receipt = _fold(view, _authored(view, note="The read confirms the goal."), tmp_path)

    assert receipt.status == "applied" and receipt.replaced_notes is None
    assert folded[:2] == before[:2]  # both notes in place, bytes, hashes and versions unchanged
    assert receipt.selected_unit_ids == (units[2].unit_id,)
    assert set(receipt.retained_unit_ids) == {units[0].unit_id, units[1].unit_id}
    assert len(folded) == 3 and "-result-tail-read" not in json.dumps(folded)
    assert folded[2]["content"][0]["text"] == (
        "[Actor-authored working view; exact source retained by checkpoint]\nThe read confirms the goal.")


@pytest.mark.parametrize("source_kind", ["restored", "emergency_body", "emergency_host_rows"])
def test_keep_selection_folds_host_sources_but_only_named_notes(tmp_path, source_kind):
    from ouroboros.context_budget import bound_host_row, authored_view_fields
    from ouroboros.context_source_view import emergency_address_view
    from ouroboros.tools.compact_context import _compact_context, record_context_view

    source = _large_unit("original")
    goal, receipt = _fold(source, _authored(source, note="Goal: keep the exact decision."), tmp_path)
    original = cc.context_units(source)[0]
    if source_kind == "restored":
        ref = {"checkpoint_ref": receipt.checkpoint_ref, "unit_id": original.unit_id,
               "raw_sha256": original.raw_sha256}
        copies, receipt = _fold(goal, replace(_authored(goal, note=""), keep_unit_ids=None,
                                             restore_unit_refs=(ref,)), tmp_path)
        host = copies[-1]
        assert "-result-tail-original" in host["content"][0]["text"]
    else:
        if source_kind == "emergency_host_rows":
            source = [{"role": "user", "content": "Owner assignment"},
                      bound_host_row("ouroboros_context_facts", "Old facts. " * 2000),
                      bound_host_row("ouroboros_context_facts", "Current facts.")]
        copies, receipt = emergency_address_view(
            source, _request(source), drive_root=tmp_path, task_id="compaction",
            rung="host_copies" if source_kind == "emergency_host_rows" else "unseen_bodies")
        host = next(row for row in copies if cc._capsule_metadata(row)[1])
    assert receipt.status == "applied"
    assert cc._capsule_metadata(host)[1]["retention"] == "source_view"
    helper = _note(tmp_path, "A helper's account remains a note.", "helper")
    helper["content"][0]["_context_capsule"]["authorship"] = "helper"
    unknown = _unattributed(_note(tmp_path, "Unrecorded authorship remains protected.", "legacy"))
    unmarked = _note(tmp_path, "Host authorship alone is not an exact source marker.", "unmarked")
    unmarked["content"][0]["_context_capsule"]["authorship"] = "host"
    notes = [*goal, helper, unknown, unmarked]
    view = [*notes, host, *_large_unit("read-b"), *_unit("recent-1"), *_unit("recent-2")]
    units = cc.context_units(view, scope="dialogue")
    host_id = units[len(notes)].unit_id

    kept, receipt = _fold(view, _authored(view, keep=(host_id,)), tmp_path)
    assert receipt.status == "applied" and host in kept and all(note in kept for note in notes)
    ctx = SimpleNamespace(active_context_mode="low")
    record_context_view(ctx, view, [])
    _compact_context(ctx, working_note="The older read is understood.", keep_last_n=2)
    counted, receipt = _fold(view, replace(_request(view), **authored_view_fields(ctx._pending_compaction)), tmp_path)
    assert receipt.status == "applied" and host in counted and all(note in counted for note in notes)
    assert [row["tool_call_id"] for row in counted if row.get("role") == "tool"] == ["recent-1", "recent-2"]

    folded, receipt = _fold(view, _authored(view, note="The new evidence agrees."), tmp_path)
    assert receipt.status == "applied" and receipt.replaced_notes is None
    assert host not in folded and all(note in folded for note in notes)
    assert receipt.selected_unit_ids == tuple(unit.unit_id for unit in units[len(notes):])
    assert folded[:len(notes)] == notes and len(folded) == len(notes) + 1
    assert "-result-tail-original" not in json.dumps(folded)


def test_named_notes_merge_into_a_new_note_that_keeps_their_restorable_addresses(tmp_path):
    """Explicit replacement: only the named note leaves; the new note and the receipt name it, and the
    address the new note carries restores the replaced note's exact text."""
    goal = _note(tmp_path, "Goal: a few elegant generalizable prompt fixes.\nDecided 05.11 with the owner.", "goal")
    other = _note(tmp_path, "An unnamed note stays.", "other")
    view = [goal, other, *_large_unit("read", "r")]
    units = cc.context_units(view, scope="dialogue")
    named = {"unit_id": units[0].unit_id, "raw_sha256": units[0].raw_sha256}

    folded, receipt = _fold(view, replace(_authored(view, note="Merged: the goal stands and the read agrees."),
                                          replace_note_ids=(named["unit_id"],)), tmp_path)

    assert receipt.status == "applied" and goal not in folded and folded[1] is other
    label, _, body = folded[0]["content"][0]["text"].partition("\n")  # the first replaced unit's place
    assert body == "Merged: the goal stands and the read agrees."
    address = json.loads(label.split("; replaced notes: ", 1)[1][:-1])
    assert address == {"checkpoint_ref": {key: receipt.checkpoint_ref[key]
                                          for key in ("kind", "root", "path", "size", "sha256")}, "notes": [named]}
    assert receipt.replaced_notes == {**address, "notes": [
        {**named, "first_line": "Goal: a few elegant generalizable prompt fixes."}]}
    assert folded[0]["content"][0]["_context_capsule"]["generation"] == 2
    observed = copy.deepcopy(folded)
    ref = {"checkpoint_ref": address["checkpoint_ref"], **named}
    restored, again = _fold(observed, replace(_authored(observed, note=""), keep_unit_ids=None,
                                              restore_unit_refs=(ref,)), tmp_path)
    assert again.status == "applied" and restored[:len(observed)] == observed
    source = restored[-1]["content"][0]["text"]
    assert cc._capsule_metadata(restored[-1])[1]["retention"] == "source_view"
    assert "Goal: a few elegant generalizable prompt fixes." in source and "Decided 05.11 with the owner." in source


@pytest.mark.parametrize("conflict", ["kept_and_named", "named_tool_unit"])
def test_a_note_both_kept_and_named_or_a_named_non_note_changes_nothing(monkeypatch, tmp_path, conflict):
    note = _note(tmp_path, "An earlier note.")
    view = [note, *_large_unit("read", "r")]
    units, before = cc.context_units(view, scope="dialogue"), copy.deepcopy(view)
    keep, named = (((units[0].unit_id,), units[0].unit_id) if conflict == "kept_and_named"
                   else ((), units[1].unit_id))
    monkeypatch.setattr(cc, "_persist_reclaim_checkpoint",
                        lambda *_a, **_k: pytest.fail("a refused request must not checkpoint"))

    rebuilt, receipt = _fold(view, replace(_authored(view, keep=keep), replace_note_ids=(named,)), tmp_path)

    assert rebuilt is view and view == before
    assert receipt.status == "binding_mismatch" and receipt.replaced_notes is None
    monkeypatch.undo()
    folded, applied = _fold(view, replace(_authored(view), replace_note_ids=(units[0].unit_id,)), tmp_path)
    assert applied.status == "applied" and note not in folded


@pytest.mark.parametrize("failure", ["none", "stale_view", "race", "capacity"])
def test_a_refused_merge_leaves_the_old_view_whole_and_names_no_replacement(tmp_path, failure):
    """CAS on the inspected revision, an arrival during the fold and the capacity check still refuse the
    whole candidate; only an applied fold replaces a named note."""
    note = _note(tmp_path, "An earlier note.")
    view = [note, *_large_unit("read", "r")]
    current, before = copy.deepcopy(view), copy.deepcopy(view)
    request = replace(_authored(view), replace_note_ids=(cc.context_units(view, scope="dialogue")[0].unit_id,))
    if failure == "stale_view":
        request = replace(request, expected_view_revision="0" * 64)

    def fit(_messages, _tools):
        if failure == "race":
            current.append({"role": "user", "content": "A new owner line arrived during the fold."})
        return {"accepted": failure != "capacity", "reason": "fixture window"}

    rebuilt, receipt = cc.compact_tool_history_llm(
        current, request=request, observed_messages=view, tool_schemas=[], fit_candidate=fit,
        drive_root=tmp_path, task_id="compaction")

    if failure == "none":
        assert receipt.status == "applied" and note not in rebuilt and receipt.replaced_notes
        return
    assert rebuilt is current and current[:len(before)] == before and note in current
    assert receipt.status == ("fit_rejected" if failure == "capacity" else "binding_mismatch")
    assert receipt.replaced_notes is None


def test_an_old_checkpoint_with_capsules_still_restores_and_no_merged_note_returns(tmp_path):
    """A fold checkpoint an earlier release wrote (its request without replace_note_ids, an unattributed
    capsule inside) restores by unit id and full hash; a later fold rewrites no kept capsule and brings
    no note merged in the past back."""
    from ouroboros.artifacts import store_actor_source_bytes

    older = _unattributed(_note(tmp_path, "A note an earlier release merged away.", "older"))
    old_view = [older, *_large_unit("before", "b")]
    unit = cc.context_units(old_view, scope="dialogue")[0]
    old_request = {key: value for key, value in asdict(_authored(old_view)).items() if key != "replace_note_ids"}
    checkpoint = store_actor_source_bytes(tmp_path, "compaction", category="context_checkpoints", source_id="old-fold",
                                          data=json.dumps({"messages": old_view, "request": old_request}).encode("utf-8"),
                                          extension="json")
    left = _note(tmp_path, "The account that old fold left.", "left")
    view = [left, *_large_unit("next", "n")]
    before = copy.deepcopy(left)

    folded, receipt = _fold(view, _authored(view, note="Next read understood."), tmp_path)

    assert receipt.status == "applied" and folded[0] == before
    assert "merged away" not in json.dumps(folded)
    ref = {"checkpoint_ref": checkpoint, "unit_id": unit.unit_id, "raw_sha256": unit.raw_sha256}
    restore = replace(_authored(folded, note=""), keep_unit_ids=None, restore_unit_refs=(ref,))
    restored, again = _fold(folded, restore, tmp_path)
    assert again.status == "applied" and restored[:len(folded)] == folded
    assert "A note an earlier release merged away." in restored[-1]["content"][0]["text"]
    wrong = replace(restore, restore_unit_refs=({**ref, "raw_sha256": "f" * 64},))
    unchanged, refused = _fold(folded, wrong, tmp_path)
    assert unchanged is folded and refused.status == "source_unavailable"
