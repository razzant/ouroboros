"""Published local folds, surviving source lineage and historical group views.

Only reviewer dispatch is stubbed. The tool boundary, source store, cold context,
Continue and new plan packet are the product's real consumers.
"""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from ouroboros import review_history_view as view
from ouroboros.artifacts import read_actor_source_bytes, task_artifact_dir_path
from ouroboros.context_compaction import context_units, unit_kind
from ouroboros.loop_round_limits import _run_authored_context_view
from ouroboros.loop_tool_execution import process_tool_results
from ouroboros.task_results import load_task_result
from ouroboros.tools.compact_context import _compact_context, record_context_view
from tests.test_main_authored_context import call
from tests.test_review_view_integration import _three, _capture, _index, harness as _harness, staged_body as _staged_body
from tests.test_plan_review_engine import _call, _control, _finding, _state, CLEAN
from tests.test_plan_dispute_history import DECK_SPEC, RATIONALE, OWNER_CHOICES

harness = _harness
staged_body = _staged_body
FIRST = "I retain the first argument and its original limited assumptions."
SECOND = "I retain the second argument separately, including its different assumptions."


def apply_local(ctx, messages, note, *, remove=(), all_units=False, **options):
    messages = deepcopy(messages)
    ctx.messages, ctx.active_context_mode = messages, "max"
    record_context_view(ctx, messages, [])
    keep = [u.unit_id for u in context_units(messages, scope="dialogue") if u.unit_id not in remove]
    args = {"working_note": note, **options}
    if remove or all_units:
        args["keep_unit_ids"] = [] if all_units else keep
    response = _compact_context(ctx, **args)
    assert "requested" in response, response
    messages.append(call("compact_context", args, "compact-local"))
    tools, fits = SimpleNamespace(_ctx=ctx), []

    def fit(candidate, schemas):
        fits.append(deepcopy(candidate))
        return {"accepted": True, "strict_bound_proven": False}

    process_tool_results([{"fn_name": "compact_context", "is_error": False,
        "tool_call_id": "compact-local", "result": response, "args_for_log": args, "trace_ref": {}}],
        messages, {"tool_calls": []}, lambda _: None, tools, fit_candidate=fit, tool_schemas=[])
    frame = SimpleNamespace(tools=tools, tool_schemas=[], fit_candidate=fit, round_idx=4,
        drive_root=ctx.drive_root, drive_logs=ctx.drive_root / "logs", task_id=ctx.task_id,
        event_queue=None, emit_progress=lambda _: None)
    after = _run_authored_context_view(messages, frame, ctx._pending_compaction, None)
    return after, ctx._context_view_receipt, fits


def actor_notes(messages):
    return [row for row in messages if row.get("role") == "assistant"
            and isinstance(row.get("content"), list)
            and (row["content"][0].get("_context_capsule") or {}).get("authorship") == "actor"]


def review_units(messages):
    return [u for u in context_units(messages, scope="dialogue")
            if view.REVIEW_HISTORY_MESSAGE_KEY in messages[u.start]]


def two_folds(ctx):
    before = _capture(ctx)
    first_unit = review_units(before)[0]
    first, receipt, _ = apply_local(ctx, before, FIRST, remove=[first_unit.unit_id])
    assert receipt["status"] == "applied"
    second_unit = review_units(first)[-1]
    second, receipt, fits = apply_local(ctx, first, SECOND, remove=[second_unit.unit_id])
    assert receipt["status"] == "applied"
    return before, first, second, second_unit, receipt, fits


def test_two_local_folds_keep_earlier_note_index_and_sources_through_cold_packet_continue(harness, monkeypatch):
    from ouroboros.context import _task_authority_projection
    from tests.test_review_view_continue import successor

    ctx, sub, _ = _three(harness, monkeypatch)
    original_state = deepcopy(_state(harness))
    before, first, second, unit, receipt, fits = two_folds(ctx)
    index_at = next(i for i, row in enumerate(before) if view.REVIEW_CONTEXT_INDEX_KEY in row)
    assert first[index_at] == second[index_at] == before[index_at]
    assert second[:unit.start] == first[:unit.start], "Final publication rewrote an unchanged early prefix"
    notes = actor_notes(second)
    assert len(notes) == 2 and actor_notes(first)[0] == notes[0]
    assert all(note in fits[-1] for note in notes)
    selected = view.load_review_history_view(receipt["selected_review_history_view"],
        lambda ref: read_actor_source_bytes(ctx.drive_root, ctx.task_id, ref))
    assert view.selected_actor_capsules(selected) == notes
    assert len(selected["covered"]) == 2
    for source in selected["covered"]:
        assert read_actor_source_bytes(ctx.drive_root, ctx.task_id, source["source_ref"])
    assert actor_notes(_capture(harness.make_ctx())) == notes
    assert _state(harness) == original_state
    # Exact paid replay stays unchanged; a genuinely new packet reads both accounts.
    _call(ctx, plan="Proofread captions and state chart units.")
    assert len(sub.calls) == 3 and _state(harness) == original_state
    _call(harness.make_ctx(), plan="Check the final accessible chart export.")
    packet = str(sub.calls[-1]["request"].messages)
    assert FIRST in packet and SECOND in packet and len(sub.calls) == 4
    env, task = successor(ctx)
    runtime = _task_authority_projection(env, task)
    historical = runtime["predecessor_authority"]["historical_review_context"]
    assert FIRST in historical["authored_account"]["text"] and SECOND in historical["authored_account"]["text"]
    assert historical["authored_account"]["authorship"] == "predecessor_actor"
    assert not historical["source_gaps"]
    assert view.SELECTED_VIEW_FIELD not in load_task_result(ctx.drive_root, task["id"])


def test_multi_note_same_note_noop_transfer_targets_the_named_account_and_explicit_merge(harness, monkeypatch):
    ctx, _, _ = _three(harness, monkeypatch)
    _, _, messages, _, _, _ = two_folds(ctx)
    notes = actor_notes(messages)
    repeated, receipt, _ = apply_local(ctx, messages, SECOND)
    assert receipt["status"] == "no_op" and actor_notes(repeated) == notes
    assert receipt["selection_fingerprint"] == view._actor_capsule(notes[1])["unit_id"].removeprefix("view:")
    options = view.attachment_transfer_options(ctx)
    transfer = {"source": options["sources"][0], "operative_spec_sha256": options["operative_spec_sha256"],
                "decision_ids": options["decision_ids"]}
    transferred, receipt, _ = apply_local(ctx, repeated, SECOND, review_transfers=[transfer])
    assert receipt["status"] == "applied"
    selected = view.load_review_history_view(receipt["selected_review_history_view"],
        lambda ref: read_actor_source_bytes(ctx.drive_root, ctx.task_id, ref))
    assert selected["capsule"] == notes[1] and view.selected_actor_capsules(selected) == notes
    assert OWNER_CHOICES not in str(_index(transferred)[-1])
    units = [u for u in context_units(transferred, scope="dialogue") if unit_kind(transferred, u) == "capsule"]
    merged, receipt, _ = apply_local(ctx, transferred, "I reconcile both accounts without losing either source.",
                                    remove=[u.unit_id for u in units])
    assert receipt["status"] == "applied" and len(actor_notes(merged)) == 1
    selected = view.load_review_history_view(receipt["selected_review_history_view"],
        lambda ref: read_actor_source_bytes(ctx.drive_root, ctx.task_id, ref))
    assert len(selected["covered"]) == 2
    assert all(ref in view._actor_capsule(actor_notes(merged)[0])["source_refs"] for ref in selected["covered"])


def test_bad_optional_transfer_keeps_its_attachment_while_valid_transfer_and_note_apply(harness):
    harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    for name, text in (("one.txt", "FIRST ATTACHMENT"), ("two.txt", "SECOND ATTACHMENT")):
        (harness.workspace / name).write_text(text, encoding="utf-8")
    assert _control(_call(ctx, {**DECK_SPEC, "evidence": ["one.txt", "two.txt"]}))["closed"]
    options = view.attachment_transfer_options(ctx)
    assert len(options["sources"]) == 2
    transfers = [{"source": source, "operative_spec_sha256": options["operative_spec_sha256"] if i == 0 else "0" * 64,
                  "decision_ids": options["decision_ids"]} for i, source in enumerate(options["sources"])]
    after, receipt, _ = apply_local(ctx, _capture(ctx), FIRST, all_units=True, review_transfers=transfers)
    assert receipt["status"] == "applied" and FIRST in str(after)
    assert len(receipt["review_transfers"]["applied"]) == len(receipt["review_transfers"]["unapplied"]) == 1
    assert "stale" in receipt["review_transfers"]["unapplied"][0]["reason"]
    current = _index(after)[-1]
    assert "FIRST ATTACHMENT" not in str(current) and "SECOND ATTACHMENT" in str(current)


def test_changed_early_decision_changes_that_index_while_tail_note_stays_local(harness, monkeypatch):
    ctx, _, _ = _three(harness, monkeypatch)
    _, first, _, _, _, _ = two_folds(ctx)
    at = next(i for i, row in enumerate(first) if view.REVIEW_CONTEXT_INDEX_KEY in row)
    choice = view.review_note_options(ctx)["entries"][0]
    after, receipt, _ = apply_local(ctx, first, SECOND,
        review_notes=[{"bound_decision": choice["bound_decision"], "remark": "An earlier alternative.", "reason": "The owner chose the existing route."}])
    assert receipt["status"] == "applied"
    assert after[:at] == first[:at] and after[at] != first[at]
    assert view.REVIEW_CONTEXT_INDEX_KEY in after[at]
    assert actor_notes(first)[0] in after


def test_grouped_closed_history_keeps_current_open_fact_new_packet_and_exact_paid_artifacts(harness, monkeypatch):
    ctx, sub, _ = _three(harness, monkeypatch)
    sub.answers = {"s1": json.dumps([_finding("current", "blocking", breaks="claim_1", summary="CURRENT OPEN REQUIREMENT", rec="Keep the current format exact.")]),
                   "s2": CLEAN, "s3": CLEAN}
    assert not _control(_call(ctx, plan="The current unfinished revision."))["closed"]
    state = deepcopy(_state(harness))
    raw = {wave["wave_artifact"]["path"]: read_actor_source_bytes(ctx.drive_root, ctx.task_id, wave["wave_artifact"])
           for wave in state["waves"]}
    observed = _capture(ctx)
    record_context_view(ctx, observed, [])
    inspection = json.loads(_compact_context(ctx, inspect=True))
    entries = inspection["review_decisions"]["entries"]
    group = [e["decision_ref"] for e in entries if e["groupable"]]
    current = [e for e in entries if not e["groupable"] and e["bound_decision"]]
    assert group and current
    after, receipt, _ = apply_local(ctx, observed, FIRST, all_units=True, review_notes=[
        {"bound_decisions": group, "remark": "Earlier presentation alternatives.", "reason": "I chose charts under the prior owner contract; new accessibility needs may reopen that choice."},
        {"bound_decisions": [current[-1]["decision_ref"]], "remark": "Do not hide this.", "reason": "An invalid optional current selection."}])
    assert receipt["status"] == "applied"
    rows = _index(after)[-1]["decision_rows"]
    assert any(r["decision_kind"] == "actor_history_group" for r in rows)
    assert not any(r.get("history_role") == "historical" for r in rows)
    assert "CURRENT OPEN REQUIREMENT" in str(after) and "Keep the current format exact." in str(after)
    assert RATIONALE not in str(after)
    assert any(g["reason"] == "current_or_unresolved_decision_stays_explicit" for g in receipt["review_notes"]["unshortened"])
    assert _state(harness) == state
    assert all(read_actor_source_bytes(ctx.drive_root, ctx.task_id, w["wave_artifact"]) == raw[w["wave_artifact"]["path"]]
               for w in state["waves"])
    cold = _capture(harness.make_ctx())
    assert "actor_history_group" in str(cold) and RATIONALE not in str(cold)
    assert "CURRENT OPEN REQUIREMENT" in str(cold)
    sub.answers = {"s1": CLEAN, "s2": CLEAN, "s3": CLEAN}
    _call(harness.make_ctx(), plan="Another current revision with a new source identity.")
    packet = str(sub.calls[-1]["request"].messages)
    assert "Earlier presentation alternatives." in packet and "CURRENT OPEN REQUIREMENT" in packet
    assert RATIONALE not in packet


def test_multiple_note_selection_loss_restores_sources_with_gap(harness, monkeypatch):
    ctx, _, _ = _three(harness, monkeypatch)
    before, _, _, _, receipt, _ = two_folds(ctx)
    ref = receipt["selected_review_history_view"]["source_ref"]
    (task_artifact_dir_path(ctx.drive_root, ctx.task_id, create=False) / ref["path"]).unlink()
    cold = _capture(harness.make_ctx())
    assert "REVIEW_HISTORY_VIEW_SOURCE_UNAVAILABLE" in str(cold)
    assert [row for row in cold if view.REVIEW_HISTORY_MESSAGE_KEY in row] == [
        row for row in before if view.REVIEW_HISTORY_MESSAGE_KEY in row]


def test_standalone_restore_keeps_selected_review_group_and_indexes(harness, monkeypatch):
    ctx, sub, _ = _three(harness, monkeypatch)
    group = [e["decision_ref"] for e in view.review_note_options(ctx)["entries"] if e["groupable"]]
    assert group
    messages, receipt, _ = apply_local(ctx, _capture(ctx), FIRST, all_units=True, review_notes=[{
        "bound_decisions": group, "remark": "Earlier presentation alternatives.",
        "reason": "I chose charts under the prior owner contract."}])
    assert receipt["status"] == "applied" and "actor_history_group" in str(_index(messages))
    # An ordinary later review appends its new index while the earlier selected
    # snapshot remains in the sent prefix. Restoration must retain both.
    ctx.messages = messages
    result = _call(ctx, plan="Check the final chart labels.")
    messages.append(call("plan_task", {}, "later-plan"))
    tools = SimpleNamespace(_ctx=ctx)
    process_tool_results([{"fn_name": "plan_task", "is_error": False,
        "tool_call_id": "later-plan", "result": str(result), "args_for_log": {}, "trace_ref": {}}],
        messages, {"tool_calls": []}, lambda _: None, tools,
        fit_candidate=lambda *_: {"accepted": True}, tool_schemas=[])
    assert len(_index(messages)) == 2
    saved = deepcopy(load_task_result(ctx.drive_root, ctx.task_id, strict=True))
    pointer = saved[view.SELECTED_VIEW_FIELD]
    selected_bytes = read_actor_source_bytes(ctx.drive_root, ctx.task_id, pointer["source_ref"])
    notes, indexes, cold = actor_notes(messages), _index(messages), _capture(harness.make_ctx())
    ref = next(ref for ref in receipt["source_refs"] if "unit_id" in ref)
    record_context_view(ctx, messages, [])
    args = {"restore_unit_refs": [ref]}
    response = _compact_context(ctx, **args)
    assert "requested" in response
    messages.append(call("compact_context", args, "restore"))
    process_tool_results([{"fn_name": "compact_context", "is_error": False,
        "tool_call_id": "restore", "result": response, "args_for_log": args, "trace_ref": {}}],
        messages, {"tool_calls": []}, lambda _: None, tools,
        fit_candidate=lambda *_: {"accepted": True}, tool_schemas=[])
    before = deepcopy(messages)
    frame = SimpleNamespace(tools=tools, tool_schemas=[], fit_candidate=lambda *_: {"accepted": True}, round_idx=5,
        drive_root=ctx.drive_root, drive_logs=ctx.drive_root / "logs", task_id=ctx.task_id,
        event_queue=None, emit_progress=lambda _: None)
    after = _run_authored_context_view(messages, frame, ctx._pending_compaction, None)
    assert ctx._context_view_receipt["status"] == "applied"
    assert not ctx._context_view_receipt["selection_fingerprint"]
    assert after[:len(before)] == before and len(after) == len(before) + 2
    assert actor_notes(after) == notes and _index(after) == indexes
    assert load_task_result(ctx.drive_root, ctx.task_id, strict=True) == saved
    assert read_actor_source_bytes(ctx.drive_root, ctx.task_id, pointer["source_ref"]) == selected_bytes
    assert _capture(harness.make_ctx()) == cold and len(sub.calls) == 4


def test_legacy_singleton_extends_to_two_surviving_accounts(harness, monkeypatch):
    from ouroboros.artifacts import store_actor_source_bytes
    from ouroboros.task_results import write_task_result
    ctx, _, _ = _three(harness, monkeypatch)
    before = _capture(ctx)
    first, receipt, _ = apply_local(ctx, before, FIRST, remove=[review_units(before)[0].unit_id])
    pointer = receipt["selected_review_history_view"]
    legacy = json.loads(read_actor_source_bytes(ctx.drive_root, ctx.task_id, pointer["source_ref"]))
    legacy.pop("capsules")
    ref = store_actor_source_bytes(ctx.drive_root, ctx.task_id, category="context_checkpoints", source_id="legacy-view",
                                   extension="json", data=json.dumps(legacy).encode("utf-8"))
    write_task_result(ctx.drive_root, ctx.task_id, "running", **{view.SELECTED_VIEW_FIELD: {**pointer, "source_ref": ref}})
    assert actor_notes(_capture(ctx)) == actor_notes(first)
    second, receipt, _ = apply_local(ctx, first, SECOND, remove=[review_units(first)[-1].unit_id])
    assert receipt["status"] == "applied" and actor_notes(_capture(ctx)) == actor_notes(second)
    assert len(actor_notes(second)) == 2


def test_compact_decision_ref_selects_only_observed_version_not_reused_position(harness, monkeypatch):
    from ouroboros.tools.plan_review import _apply_disposition
    ctx, _, _ = _three(harness, monkeypatch)
    observed = _capture(ctx)
    entry = view.review_note_options(ctx)["entries"][-1]
    wave = _state(harness)["waves"][-1]
    assert entry["finding_id"]
    _apply_disposition(ctx, {"review_fingerprint": wave["request_fingerprint"], "items": [
        {"finding_id": entry["finding_id"], "decision": "reject", "rationale": "NEW SOURCE-BOUND REASON"}]})
    after, receipt, _ = apply_local(ctx, observed, FIRST, all_units=True, review_notes=[
        {"bound_decision": entry["decision_ref"], "remark": "Stale source.", "reason": "Do not apply."},
        {"bound_decision": "decision_1", "remark": "Position alone.", "reason": "Do not apply."}])
    assert receipt["status"] == "applied" and FIRST in str(after)
    assert [n["bound_decision"] for n in receipt["review_notes"]["applied"]] == [entry["bound_decision"]]
    assert "NEW SOURCE-BOUND REASON" in str(_index(after)[-1])
    assert any(g["reason"] == "Invalid review body binding" for g in receipt["review_notes"]["unshortened"])


def test_obsolete_index_helper_keeps_latest_and_unknown_rows():
    first = view._index_message({"value": 1}, "plan")
    current = view._index_message({"value": 2}, "plan")
    other = view._index_message({"value": 3}, "commit", "/another/repo")
    invalid = deepcopy(current)
    invalid["content"] += "not the bound index"
    assert view.obsolete_review_index_positions([first, other, current, invalid]) == (0,)
    assert view.obsolete_review_index_positions([current, invalid]) == ()


def test_old_settled_fail_and_linked_open_obligation_survive_another_history_group(staged_body, tmp_path, monkeypatch):
    from pathlib import Path
    from ouroboros import capability_evidence, reviewer_window, review_substrate, review_ledger
    from ouroboros.review_history import review_dispute_history
    from ouroboros.review_state import load_state, save_state, ObligationItem, make_repo_key
    from ouroboros.tools.registry import ToolContext
    from tests.test_review_cold_history import _run, _dispatch
    from tests import _contributor_packet_shared as shared

    monkeypatch.setattr(capability_evidence, "probe", lambda *a, **kw: None)
    monkeypatch.setattr(reviewer_window, "reviewer_context_window", lambda *a, **kw: 1_000_000)
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "6")
    root, drive = Path(staged_body["repo"]), tmp_path / "data"
    ctx = ToolContext(repo_dir=root, drive_root=drive, task_id="history-group")
    records = []
    for version in range(3):
        (root / "ouroboros/tools/review.py").write_text(f"RULES = 'group fixture {version}'\n", encoding="utf-8")
        shared.git(root, "add", "-A")
        dispatch = _dispatch([], "OPEN CRITIC REASON" if version == 0 else f"Resolved historical argument {version}")

        def serve(request, **kwargs):
            result = dispatch(request, **kwargs)
            if version == 0:
                for actor in result.actors:
                    answer = json.loads(actor["raw_text"])
                    items = answer if isinstance(answer, list) else answer["change"]
                    if items:
                        items[0].update(verdict="FAIL", severity="critical")
                    actor["raw_text"] = json.dumps(answer)
            return result

        monkeypatch.setattr(review_substrate, "run_review_request", serve)
        records.append(_run(ctx, "change", ""))
    assert review_ledger.load_record(drive, records[0])["verdict"]["aggregate"] == "FAIL"
    # The obligation owner retains unresolved work independently of later paid
    # review settlement. Use its persisted shape and exact attempt association.
    state = load_state(drive)
    state.open_obligations.append(ObligationItem("obl-closed-history-test", "parser", "critical",
        "OPEN OBLIGATION STILL NEEDS REPAIR", "2026-10-10T00:00:00Z", "prior failed review",
        repo_key=make_repo_key(root)))
    next(a for a in state.attempts if a.review_record_id == records[0]).obligation_ids.append("obl-closed-history-test")
    save_state(drive, state)
    canonical = [deepcopy(review_ledger.load_record(drive, rid)) for rid in records]
    options = view.review_note_options(ctx)["entries"]
    group = [e["bound_decision"] for e in options if e["groupable"]]
    assert group
    after, receipt, _ = apply_local(ctx, _capture(ctx), FIRST, all_units=True, review_notes=[
        {"bound_decisions": group, "remark": "The closed experiment.", "reason": "I retained its useful conclusion separately from the unresolved parser."}])
    assert receipt["status"] == "applied"
    shown = _index(after)[-1]
    assert "OPEN CRITIC REASON" in str(shown) and "OPEN OBLIGATION STILL NEEDS REPAIR" in str(shown)
    assert any(r["decision_kind"] == "actor_history_group" for r in shown["decision_rows"])
    prior = next(r for r in shown["rounds"] if r.get("review_record_id") == records[0])
    assert prior["verdict"]["aggregate"] == "FAIL"
    assert [review_ledger.load_record(drive, rid) for rid in records] == canonical
    assert review_dispute_history(drive_root=drive, repo_root=root, task_id=ctx.task_id)["open_obligations"]


def _groups(messages):
    return [row for row in _index(messages)[-1]["decision_rows"] if row["decision_kind"] == "actor_history_group"]


def _tail_fold(ctx, messages, note):
    before = deepcopy(messages)
    before.extend([call("read_file", {"path": "unrelated.txt"}, "unrelated-tail"),
                   {"role": "tool", "tool_call_id": "unrelated-tail", "content": "Unrelated measured tail. " * 1000}])
    unit = context_units(before, scope="dialogue")[-1]
    after, receipt, _ = apply_local(ctx, before, note, remove=[unit.unit_id])
    assert receipt["status"] == "applied"
    assert after[:unit.start] == before[:unit.start], "An unchanged earlier group rewrote the prefix"
    return after, receipt


def test_explicit_larger_group_retires_covered_groups_preserves_disjoint_and_exact_old_source(harness, monkeypatch):
    ctx, _, _ = _three(harness, monkeypatch)
    bindings = [e["bound_decision"] for e in view.review_note_options(ctx)["entries"] if e["groupable"]]
    assert len(bindings) >= 3
    notes = [{"bound_decisions": [binding], "remark": f"Account {i}", "reason": f"Exact earlier reason {i}."}
             for i, binding in enumerate(bindings[:3])]
    before, first, _ = apply_local(ctx, _capture(ctx), FIRST, all_units=True, review_notes=notes)
    source = first["selected_review_history_view"]["source_ref"]
    original = read_actor_source_bytes(ctx.drive_root, ctx.task_id, source)
    merged, receipt, _ = apply_local(ctx, before, SECOND, all_units=True, review_notes=[
        {"bound_decisions": bindings[:2], "remark": "Merged account", "reason": "The newer account replaces both earlier reasons."}])
    groups = _groups(merged)
    assert len(groups) == 2 and {g["remark"] for g in groups} == {"Account 2", "Merged account"}
    assert not any(g["reason"] in {notes[0]["reason"], notes[1]["reason"]} for g in groups)
    disjoint = next(g for g in groups if g["remark"] == "Account 2")
    assert disjoint["source"] == {"source_ref": source, "field": "review_notes[2]"}
    assert read_actor_source_bytes(ctx.drive_root, ctx.task_id, source) == original
    assert _groups(_capture(ctx)) == groups
    saved = view.load_review_history_view(receipt["selected_review_history_view"],
        lambda ref: read_actor_source_bytes(ctx.drive_root, ctx.task_id, ref))
    assert len(saved["review_notes"]) == 2


def test_unchanged_group_source_and_final_prefix_survive_two_unrelated_tail_folds(harness, monkeypatch):
    ctx, _, _ = _three(harness, monkeypatch)
    bindings = [e["bound_decision"] for e in view.review_note_options(ctx)["entries"] if e["groupable"]]
    grouped, first, _ = apply_local(ctx, _capture(ctx), FIRST, all_units=True, review_notes=[
        {"bound_decisions": bindings, "remark": "Closed history", "reason": "The accepted premise may be revisited when requirements change."}])
    original_group = deepcopy(_groups(grouped)[0])
    original_source = first["selected_review_history_view"]["source_ref"]
    raw = read_actor_source_bytes(ctx.drive_root, ctx.task_id, original_source)
    for index in range(2):
        grouped, receipt = _tail_fold(ctx, grouped, f"Independent tail account {index}.")
        assert _groups(grouped) == [original_group]
        assert receipt["selected_review_history_view"]["source_ref"] != original_source
    assert len(actor_notes(grouped)) == 3
    assert _groups(_capture(ctx)) == [original_group]
    assert read_actor_source_bytes(ctx.drive_root, ctx.task_id, original_source) == raw
    reads = []
    def reader(ref):
        reads.append(ref["sha256"])
        return read_actor_source_bytes(ctx.drive_root, ctx.task_id, ref)
    view.load_review_history_view(receipt["selected_review_history_view"], reader)
    assert original_source["sha256"] in reads, "The anchored source must be verified by the existing exact reader"


def test_changed_group_prose_gets_new_source_and_model_cannot_supply_old_anchor(harness, monkeypatch):
    from ouroboros.artifacts import store_actor_source_bytes
    ctx, _, _ = _three(harness, monkeypatch)
    bindings = [e["bound_decision"] for e in view.review_note_options(ctx)["entries"] if e["groupable"]]
    note = {"bound_decisions": bindings, "remark": "The same members", "reason": "My earlier understanding."}
    first, receipt, _ = apply_local(ctx, _capture(ctx), FIRST, all_units=True, review_notes=[note])
    old_source = receipt["selected_review_history_view"]["source_ref"]
    old_bytes = read_actor_source_bytes(ctx.drive_root, ctx.task_id, old_source)
    anchor = {"source_ref": old_source, "note_index": 0}
    changed = {**note, "reason": "The corrected understanding changes the earlier explanation.", "account_source": anchor}
    revised, receipt, _ = apply_local(ctx, first, SECOND, all_units=True, review_notes=[changed])
    pointer = receipt["selected_review_history_view"]
    group = _groups(revised)[0]
    assert group["reason"] == changed["reason"] and group["source"]["source_ref"] == pointer["source_ref"]
    assert group["source"]["source_ref"] != old_source
    assert _groups(_capture(ctx)) == [group]
    assert read_actor_source_bytes(ctx.drive_root, ctx.task_id, old_source) == old_bytes
    saved = view.load_review_history_view(pointer, lambda ref: read_actor_source_bytes(ctx.drive_root, ctx.task_id, ref))
    assert "account_source" not in saved["review_notes"][0]
    # Cold parsing rejects a source with the same membership but another actor text.
    saved["review_notes"][0]["account_source"] = anchor
    invalid_ref = store_actor_source_bytes(ctx.drive_root, ctx.task_id, category="context_checkpoints",
        source_id="mismatched-group-source", extension="json", data=json.dumps(saved).encode("utf-8"))
    with pytest.raises(ValueError, match="exact authored payload"):
        view.load_review_history_view({**pointer, "source_ref": invalid_ref},
            lambda ref: read_actor_source_bytes(ctx.drive_root, ctx.task_id, ref))
