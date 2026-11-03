"""Review history producers, authored materializer, task selection and new readers.

The plan family reads the records the retired plan-review engine left (recorded from its
real flows, ``tests/fixtures/legacy_plan_dispute``); the commit/change family runs its real
producers with only the review model dispatch a local fixture. Nothing calls a provider.
"""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from ouroboros import review_history_view as view
from ouroboros.artifacts import read_actor_source_bytes, task_artifact_dir_path
from ouroboros.context import _task_authority_projection
from ouroboros.loop_round_limits import _run_authored_context_view
from ouroboros.loop_tool_execution import process_tool_results
from ouroboros.task_results import load_task_result
from ouroboros.tools.compact_context import _compact_context, record_context_view
from tests import _legacy_plan_shared as legacy
from tests.test_plan_dispute_history import (
    _first_two_rounds, _state, harness as _plan_harness, LATE_FEEDBACK, RATIONALE, OWNER_CHOICES, RAW_REASON,
)
from tests.test_review_history_view import same_decisions

harness = _plan_harness
from tests.test_main_authored_context import call

NOTE = "I retain charts. The owner authorized no new service; the rejected tables recommendation remains attributed evidence."


def _capture(ctx):
    runtime = _task_authority_projection(SimpleNamespace(drive_root=ctx.drive_root, repo_dir=ctx.repo_dir), {"id": ctx.task_id})
    prefix, tail = view.capture_review_history_messages(runtime, task_id=ctx.task_id, drive_root=ctx.drive_root)
    return [{"role": "system", "content": json.dumps(prefix)}, {"role": "user", "content": "Finish the authorized deck."}, *tail]


def _apply(ctx, messages, *, note=NOTE, transfers=(), review_notes=(), expected_view_revision="", fit=None):
    """Actual tool observation, complete tool batch, materialization and publication."""
    schemas, observed_fits = [], []
    def measure(candidate, tools):
        observed_fits.append(deepcopy(candidate))
        return fit(candidate, tools) if fit else {"accepted": True, "strict_bound_proven": False}
    tools = SimpleNamespace(_ctx=ctx)
    ctx.messages = messages
    ctx.active_context_mode = "max"
    record_context_view(ctx, messages, schemas)
    response = _compact_context(ctx, working_note=note, keep_unit_ids=[], review_transfers=list(transfers),
                                review_notes=list(review_notes), expected_view_revision=expected_view_revision)
    assert "requested" in response, response
    messages.append(call("compact_context", {"working_note": note, "keep_unit_ids": [], "review_notes": list(review_notes), "expected_view_revision": expected_view_revision}, "compact"))
    execution = {"fn_name": "compact_context", "is_error": False, "tool_call_id": "compact",
                 "result": response, "args_for_log": {"working_note": note}, "trace_ref": {}}
    process_tool_results([execution], messages, {"tool_calls": []}, lambda _: None, tools,
                         fit_candidate=measure, tool_schemas=schemas)
    frame = SimpleNamespace(tools=tools, tool_schemas=schemas, fit_candidate=measure, round_idx=4,
        drive_root=ctx.drive_root, drive_logs=ctx.drive_root / "logs", task_id=ctx.task_id,
        event_queue=None, emit_progress=lambda _: None)
    after = _run_authored_context_view(messages, frame, ctx._pending_compaction, None)
    return after, ctx._context_view_receipt, observed_fits


def _three(harness):
    """The retired engine's three closed note rounds, as recorded (``three_rounds``): two_rounds,
    then round 3 with s1 note 'units' ('State chart units'). Returns the context and s1's
    round-1 answer."""
    meta = legacy.plant_recorded(harness.drive, "three_rounds", workspace=harness.workspace)
    return harness.make_ctx(), meta["raw"]


def _index(messages):
    return [json.loads(row["content"].partition("\n")[2])["current"]
            for row in messages if view.REVIEW_CONTEXT_INDEX_KEY in row]


def test_three_waves_actor_note_cold_author_and_later_round_keep_current_index(harness, monkeypatch):
    ctx, raw = _three(harness)
    messages = _capture(ctx)
    history, operative = view.current_plan_history(ctx)
    assert operative["spec"]
    assert raw in [b["value"] for b in view.split_review_history(history)["bodies"]]
    before_state = deepcopy(_state(harness))
    after, receipt, fits = _apply(ctx, messages)
    assert receipt["status"] == "applied", receipt
    saved = load_task_result(ctx.drive_root, ctx.task_id, strict=True)
    pointer = saved[view.SELECTED_VIEW_FIELD]
    selected = view.load_review_history_view(pointer, lambda ref: read_actor_source_bytes(ctx.drive_root, ctx.task_id, ref))
    capsule = selected["capsule"]
    assert capsule in after and NOTE in str(capsule)
    assert selected["covered"]
    assert raw not in [b.get("value") for b in view.selected_review_history(history,
        drive_root=ctx.drive_root, task_id=ctx.task_id, operative_subject=operative)["bodies"]]
    assert RAW_REASON not in str(after), [(i, row.get("role"), list(row), str(row.get("content"))[:120]) for i, row in enumerate(after) if RAW_REASON in str(row)]
    assert OWNER_CHOICES in str(after), "attachment absent after ordinary note"
    assert same_decisions(_index(after)[-1], history)
    assert _index(after)[-1]["operative_subject"]["spec"] == operative["spec"]
    checkpoint = json.loads(read_actor_source_bytes(ctx.drive_root, ctx.task_id, receipt["checkpoint_ref"]))
    assert RAW_REASON in str(checkpoint["messages"])
    assert any(view.REVIEW_CONTEXT_INDEX_KEY in row for row in fits[-1])
    assert _state(harness) == before_state  # note is not a plan attempt or a paid cycle
    from tests.test_review_history_context import core, plan_for
    plan = plan_for(core(extra=checkpoint["messages"][2:]), monkeypatch, data_root=ctx.drive_root)
    rebound = plan.reproject_for_route(window_tokens=900_000, known_window=True, ratio=1.0,
        output_reserve=65_536, tool_schemas=[], current_messages=after)
    for mode in ("max", "low", "nano"):
        assert rebound.messages_for(mode)[1:] == after[1:]
        assert RAW_REASON not in str(rebound.messages_for(mode))
    cold = _capture(harness.make_ctx())
    assert capsule in cold and RAW_REASON not in str(cold) and OWNER_CHOICES in str(cold)
    assert same_decisions(_index(cold)[-1], history)
    # A later recorded round lands after the selection (a cold author's closed round 4,
    # s1 note 'late'): its body is not covered by the selection and stays in the view.
    legacy.advance_recorded(ctx.drive_root, "three_rounds_then_late_note", workspace=harness.workspace)
    new_history, new_operative = view.current_plan_history(ctx)
    newer = view.selected_review_history(new_history, drive_root=ctx.drive_root, task_id=ctx.task_id, operative_subject=new_operative)
    assert any("New uncovered requirement" in str(body["value"]) for body in newer["bodies"])


def test_explicit_attachment_transfer_requires_current_spec_and_keeps_ordinary_note_full(harness):
    ctx, _ = _three(harness)
    after, receipt, _ = _apply(ctx, _capture(ctx))
    assert receipt["status"] == "applied" and OWNER_CHOICES in str(after)
    options = view.attachment_transfer_options(ctx)
    assert options["sources"] and options["decision_ids"]
    transfer = {"source": options["sources"][0], "operative_spec_sha256": options["operative_spec_sha256"],
                "decision_ids": options["decision_ids"]}
    # Same model's new declaration uses the already authored current spec; no
    # fabricated new plan identity or semantic-coverage checker is involved.
    shorter, receipt, _ = _apply(ctx, after, note=NOTE, transfers=[transfer])
    assert receipt["status"] == "applied", receipt
    assert OWNER_CHOICES not in str(_index(shorter)[-1])
    assert "author_transferred_to_operative_spec" in str(_index(shorter)[-1])
    assert OWNER_CHOICES not in str(_capture(harness.make_ctx()))
    history, operative = view.current_plan_history(ctx)
    changed = deepcopy(operative)
    changed["spec"]["decisions"][0]["why"] = "Changed owner choice"
    restored = view.selected_review_history(history, drive_root=ctx.drive_root, task_id=ctx.task_id, operative_subject=changed)
    assert OWNER_CHOICES in str(restored["mandatory"])
    stale = {**transfer, "operative_spec_sha256": "0" * 64}
    with pytest.raises(ValueError):
        view.validate_attachment_transfers(history, operative, [stale])


@pytest.mark.parametrize("failure", ["fit", "save"])
def test_rejected_fit_or_task_selection_save_restores_raw_without_false_durability(harness, monkeypatch, failure):
    ctx, _ = _three(harness)
    messages = _capture(ctx)
    if failure == "save":
        def fail(*a, **kw):
            raise OSError("synthetic task selection write failure")
        monkeypatch.setattr(view, "publish_review_history_view", fail)
    fit = (lambda rows, tools: {"accepted": not any("[Context view receipt]" in str(row) for row in rows)}) if failure == "fit" else None
    after, receipt, _ = _apply(ctx, messages, fit=fit)
    assert receipt["status"] == ("fit_rejected" if failure == "fit" else "selection_failed")
    assert RAW_REASON in str(after)
    assert view.SELECTED_VIEW_FIELD not in load_task_result(ctx.drive_root, ctx.task_id, strict=True)


def test_lost_selected_source_restores_full_and_names_gap_without_buying_review(harness):
    ctx, _ = _three(harness)
    sub = harness.install({})
    _, receipt, _ = _apply(ctx, _capture(ctx))
    ref = receipt["selected_review_history_view"]["source_ref"]
    (task_artifact_dir_path(ctx.drive_root, ctx.task_id, create=False) / ref["path"]).unlink()
    cold = _capture(harness.make_ctx())
    assert RAW_REASON in str(cold) and "REVIEW_HISTORY_VIEW_SOURCE_UNAVAILABLE" in str(cold)
    assert _index(cold)[-1]["status"] == "source_unavailable"
    assert sub.calls == []


def test_fresh_producer_index_is_counted_before_batch_allocation_and_delivered_after_pair(harness):
    ctx, _ = _first_two_rounds(harness)
    messages = _capture(ctx)
    ctx._pending_review_context = {}
    # The next recorded round lands (s1 note 'units': "State chart units") and its producer
    # notifies the plan family, as every plan_task round did.
    result = legacy.advance_recorded(ctx.drive_root, "three_rounds", workspace=harness.workspace)["round3_result"]
    view.queue_review_history_context(ctx, family="plan")
    assert ctx._pending_review_context == {"plan": ""}
    observed = []
    def measure(rows, schemas):
        observed.append(deepcopy(rows))
        return {"accepted": True, "estimated_input_tokens": 10, "response_reserve_tokens": 1,
                "capacity_total_tokens": 100_000, "strict_bound_proven": False}
    messages.append(call("plan_task", {}, "plan-result"))
    result_row = {"fn_name": "plan_task", "is_error": False, "tool_call_id": "plan-result",
                  "result": str(result), "args_for_log": {}, "trace_ref": {}}
    process_tool_results([result_row], messages, {"tool_calls": []}, lambda _: None, SimpleNamespace(_ctx=ctx),
                         fit_candidate=measure, tool_schemas=[])
    assert observed and all("State chart units" in str(rows) for rows in observed)
    last_pair = next(i for i, row in enumerate(messages) if row.get("tool_call_id") == "plan-result")
    assert messages[last_pair - 1]["tool_calls"][0]["id"] == "plan-result"
    assert all(row.get("role") != "tool" for row in messages[last_pair + 1:])
    assert "State chart units" in str(messages[last_pair + 1:])
    assert ctx._pending_review_context == {}
    # Park/resume serializes these canonical rows, not an ephemeral side queue.
    restored = json.loads(json.dumps(messages))
    assert "State chart units" in str(_index(restored)[-1])


from tests.test_review_cold_history import staged_body as _staged_body, _run, _dispatch  # noqa: E402

staged_body = _staged_body


@pytest.mark.parametrize("surface", ["commit", "change"])
def test_commit_and_change_use_same_authored_selection_and_cold_packet(staged_body, tmp_path, monkeypatch, surface):
    from pathlib import Path
    from ouroboros import capability_evidence, reviewer_window, review_substrate, review_ledger
    from ouroboros.review_history import review_dispute_history
    from ouroboros.tools.registry import ToolContext
    from tests import _contributor_packet_shared as shared
    from tests.test_review_change_end_to_end import _brief_text

    monkeypatch.setattr(capability_evidence, "probe", lambda *a, **k: None)
    monkeypatch.setattr(reviewer_window, "reviewer_context_window", lambda *a, **k: 1_000_000)
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "6")
    root, drive = Path(staged_body["repo"]), tmp_path / "authored-review"
    records = []
    for version in range(3):
        ctx = ToolContext(repo_dir=root, drive_root=drive, task_id="shared-review-view")
        (root / "ouroboros/tools/review.py").write_text(f"RULES = 'authored version {version}'\n", encoding="utf-8")
        shared.git(root, "add", "-A")
        monkeypatch.setattr(review_substrate, "run_review_request", _dispatch([], f"Argument for version {version}"))
        records.append(_run(ctx, surface, "Keep the original reviewed rationale."))
    review_ledger.note_author_decision(drive, records[0], {"disposition": "rejected", "rationale": RATIONALE})
    history = review_dispute_history(drive_root=drive, repo_root=root, task_id=ctx.task_id)
    assert len(history["rounds"]) == 3
    after, receipt, _ = _apply(ctx, _capture(ctx))
    assert receipt["status"] == "applied", receipt
    assert view.SELECTED_VIEW_FIELD in load_task_result(drive, ctx.task_id, strict=True)
    assert same_decisions(_index(after)[-1], history)
    cold = _capture(ToolContext(repo_dir=root, drive_root=drive, task_id=ctx.task_id))
    assert NOTE in str(cold) and RATIONALE in str(cold)
    assert not [row for row in cold if view.REVIEW_HISTORY_MESSAGE_KEY in row]
    briefs = []
    monkeypatch.setattr(review_substrate, "run_review_request", _dispatch(briefs, "New fourth version"))
    (root / "ouroboros/tools/review.py").write_text("RULES = 'fourth authored version'\n", encoding="utf-8")
    shared.git(root, "add", "-A")
    _run(ToolContext(repo_dir=root, drive_root=drive, task_id=ctx.task_id), surface, "New argument remains full.")
    assert briefs
    for brief in briefs:
        text = _brief_text(brief)
        assert NOTE in text and RATIONALE in text
        assert "actor_authored_view" in text


GATE_FINDING = "The new helper drops the error path the caller relies on. GATE_FAIL_FINDING_END"


def _failing_dispatch():
    from tests import _contributor_packet_shared as shared_packet

    golden = shared_packet.golden_substrate([])

    def run(request, **kwargs):
        result = golden(request, **kwargs)
        for actor in result.actors:
            value = json.loads(actor["raw_text"])
            change = value if isinstance(value, list) else value["change"]
            (change or value["coupling"])[0].update(verdict="FAIL", severity="critical", reason=GATE_FINDING)
            actor["raw_text"] = json.dumps(value, ensure_ascii=False)
        return result

    return run


@pytest.mark.parametrize("dispatch", ["fail", "crash"])
def test_first_commit_gate_wave_reaches_the_next_request_after_the_authors_fold(
        staged_body, tmp_path, monkeypatch, dispatch):
    """#1723: the commit panel queues the same review view review_change does.

    The first wave of the task, with no resident index yet: its raw return is
    the only other carrier, and the author's fold removes it. The caller's own
    outcome is unchanged; a crashed wave queues the view and invents no verdict.
    """
    from pathlib import Path

    from ouroboros import capability_evidence, review_ledger, review_substrate, reviewer_window
    from ouroboros.tools.git_review_cycle import _run_non_committing_review_cycle
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setattr(capability_evidence, "probe", lambda *a, **k: None)
    monkeypatch.setattr(reviewer_window, "reviewer_context_window", lambda *a, **k: 1_000_000)
    root, drive = Path(staged_body["repo"]), tmp_path / "gate-view"
    ctx = ToolContext(repo_dir=root, drive_root=drive, task_id="gate-view")

    def crash(*_args, **_kwargs):
        raise RuntimeError("reviewer transport crashed")

    monkeypatch.setattr(review_substrate, "run_review_request", _failing_dispatch() if dispatch == "fail" else crash)
    outcome = _run_non_committing_review_cycle(ctx, "fix: drop the redundant helper branch",
                                               skip_advisory_review=True, goal="Ship the bounded file design")
    assert outcome["status"] != "passed", outcome
    record = review_ledger.load_record(drive, outcome["review_record_id"])
    assert record["verdict"]["aggregate"] == ("FAIL" if dispatch == "fail" else "NOT_PERFORMED"), record["verdict"]
    messages = [{"role": "system", "content": "Runtime."}, {"role": "user", "content": "Ship the bounded file design."}]
    assert not _index(messages)  # no resident index precedes this first wave
    messages.append(call("commit_reviewed", {"commit_message": "fix: drop the redundant helper branch"}, "gate"))
    raw_return = json.dumps({"raw_gate_return": "RAW_GATE_RETURN_END", **outcome}, default=str)
    process_tool_results([{"fn_name": "commit_reviewed", "is_error": False, "tool_call_id": "gate",
                           "result": raw_return, "args_for_log": {}, "trace_ref": {}}],
                         messages, {"tool_calls": []}, lambda _: None, SimpleNamespace(_ctx=ctx),
                         fit_candidate=lambda rows, schemas: {"accepted": True, "strict_bound_proven": False},
                         tool_schemas=[])
    after, receipt, fits = _apply(ctx, messages)
    assert receipt["status"] == "applied", receipt
    assert "RAW_GATE_RETURN_END" not in str(after)  # the raw return was folded away
    if dispatch == "fail":
        assert GATE_FINDING in str(after) and GATE_FINDING in str(fits[-1])  # the next model send
        cold = _capture(ToolContext(repo_dir=root, drive_root=drive, task_id=ctx.task_id))
        assert GATE_FINDING in str(cold)
    # The producer's typed notification was consumed once at the tool boundary.
    assert getattr(ctx, "_pending_review_context", None) == {}
    for index in (_index(messages)[-1], _index(after)[-1]):
        # The record itself is behind its one address; the entry names the recorded verdict, never an invented one.
        (entry,) = [e for e in index["addresses"].values() if e["read"].get("tool") == "query_data"]
        assert entry["read"]["arguments"] == {"op": "json", "kind": "review_ledger", "ids": [record["record_id"]]}
        assert f"aggregate verdict {record['verdict']['aggregate']}" in entry["what"] and "rounds" not in index


def test_late_supplement_after_selection_keeps_new_text_and_old_verdict(tmp_path):
    from ouroboros.task_results import load_plan_review_state
    from ouroboros.tools.registry import ToolContext

    task = "history-task"
    # A closed wave whose two slots were still in flight, as recorded.
    original = legacy.plant_recorded(tmp_path, "closed_wave_late_pending")["original"]
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path, task_id=task)
    after, receipt, _ = _apply(ctx, _capture(ctx))
    assert receipt["status"] == "applied"
    assert view.SELECTED_VIEW_FIELD in load_task_result(tmp_path, task, strict=True)
    # Then the recorded late answer of slot s0, attached as a historical supplement.
    late = legacy.advance_recorded(tmp_path, "closed_wave_late_supplement")
    history, operative = view.current_plan_history(ctx)
    projected = view.selected_review_history(history, drive_root=tmp_path, task_id=task, operative_subject=operative)
    assert any(body["value"] == LATE_FEEDBACK for body in projected["bodies"])
    assert projected["mandatory"]["rounds"][-1]["aggregate"] == original["aggregate"]
    cold = _capture(ctx)
    assert any(LATE_FEEDBACK in row["content"] or json.dumps(LATE_FEEDBACK)[1:-1] in row["content"] for row in cold if view.REVIEW_HISTORY_MESSAGE_KEY in row)
    assert load_plan_review_state(tmp_path, task)["waves"][-1]["historical_supplements"][0]["operation_id"] == late["operation_id"]


from tests.test_main_authored_context import main_loop as _main_loop  # noqa: E402

main_loop = _main_loop


def test_main_physical_observation_selects_typed_review_body_and_keeps_resident_index(main_loop):
    f = main_loop
    f.ctx.task_id = "authored-main"
    raw = "Complete critic reasoning, still attributed. " * 180 + "RAW_CRITIC_END"
    spec = {"goal": "Keep the authored plan", "decisions": [{"choice": "A", "rejected": ["B"], "why": "Owner's reason"}]}
    # One closed earlier-schema wave as the task's only plan record.
    wave = legacy.with_artifact(f.ctx.drive_root, f.ctx.task_id, legacy.wave(
        1, fingerprint="a" * 64, spec=spec, aggregate="GREEN", closed=True, paid=False,
        reviewer_outputs=[{"slot_id": "s1", "text": raw}]))
    legacy.seed_plan_review_state(f.ctx.drive_root, f.ctx.task_id, legacy.v2_state(wave))
    f.messages.extend(_capture(f.ctx)[2:])
    f.run([call("compact_context", {"working_note": NOTE, "keep_unit_ids": []}, "compact"), {"content": "done"}])
    assert f.ctx._context_view_receipt["status"] == "applied"
    assert view.SELECTED_VIEW_FIELD in load_task_result(f.ctx.drive_root, f.ctx.task_id, strict=True)
    assert raw not in str(f.inputs[-1]["messages"])
    assert NOTE in str(f.inputs[-1]["messages"])
    assert _index(f.ctx.messages)[-1]["operative_subject"]["spec"] == spec
