"""Full disputes survive cold author/reviewer preparation and exact-source loss."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from ouroboros.artifacts import task_artifact_dir_path
from ouroboros.context import _task_authority_projection
from ouroboros.tools import plan_review, plan_review_artifacts as artifacts, plan_spec
from tests.test_plan_review_engine import (
    CLEAN, DECK_SPEC, _call, _control, _finding, _slots, _state, _user_text,
    harness as _engine_harness,
)

harness = _engine_harness
RATIONALE = "Reject tables: the owner chose charts for the visual briefing; no new storage service is authorized."
OWNER_CHOICES = "A: charts only; B: tables plus a new service. Owner: A. Authorization is limited to the deck."
RAW_REASON = "The recommendation also avoids duplicate interpretation. " * 120 + "RAW_REASON_END"


def _history(h, ctx):
    return _task_authority_projection(SimpleNamespace(drive_root=h.drive), {"id": ctx.task_id})["plan_review_authority"]


def _resident_reason(history, alias):
    assert alias["representation"] == "resident_review_decision"
    return next(row["reason"] for row in history["decision_rows"] if row.get("decision_ref") == alias["decision_ref"])


def _first_two_rounds(h, monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "6")
    h.workspace.joinpath("notes.md").write_text(OWNER_CHOICES, encoding="utf-8")
    recommendation = "Consider tables. " + "Detailed tradeoff. " * 80 + "RECOMMENDATION_END"
    raw = json.dumps([_finding("tables", "note", summary="Use tables instead of charts", rec=recommendation)]) + "\n" + RAW_REASON
    sub = h.install({"s1": raw, "s2": CLEAN, "s3": CLEAN})
    ctx = h.make_ctx()
    first_spec = {**DECK_SPEC, "evidence": ["notes.md"]}
    assert _control(_call(ctx, first_spec)) == {"outcome": "GREEN", "closed": True}
    first = _state(h)["waves"][-1]
    plan_review._handle_plan_task(ctx, review_disposition={
        "review_fingerprint": first["request_fingerprint"],
        "items": [{"finding_id": "s1:tables", "decision": "reject", "rationale": RATIONALE}],
    })
    sub.answers = {"s1": json.dumps([_finding("captions", "note", summary="Check caption spelling")]),
                   "s2": CLEAN, "s3": CLEAN}
    assert _control(_call(ctx, plan="Proofread captions before drawing."))["closed"]
    return ctx, sub, first, raw, recommendation


@pytest.mark.parametrize("delivery", ["packet", "native", "session"])
def test_three_round_dispute_reaches_cold_author_and_new_reviewer(harness, monkeypatch, delivery):
    from ouroboros import reviewer_window
    from ouroboros.review_execution import ReviewAssignment
    from ouroboros.review_native_episode import NativeToolRoundReviewExecutor

    monkeypatch.setattr(reviewer_window, "reviewer_context_window", lambda *a, **k: 1_000_000)
    ctx, sub, first, raw, recommendation = _first_two_rounds(harness, monkeypatch)
    before = _state(harness)
    author = _history(harness, ctx)
    dispute = author["dispute_history"]
    assert dispute["status"] == "complete"
    assert any(RATIONALE in _resident_reason(dispute, d["authored_view"])["author_rationales"]
               for r in dispute["rounds"] for d in r.get("dispositions", []))
    assert any(r.get("text") == raw for wave in dispute["rounds"] for r in wave["reviewer_outputs"])
    assert "session_task" not in json.dumps(dispute)
    assert "request_messages" not in json.dumps(dispute)
    assert "reviewer_outputs" not in author["current_wave"]
    # A cold context and a changed roster cannot rely on a sticky API transcript.
    cold = harness.make_ctx()
    slots = _slots(("fresh", "m/new", "session")) if delivery == "session" else _slots(("fresh", "m/new"))
    if delivery == "native":
        slots = [replace(slots[0], native_retrieval_override=True, subagent_id="new-reader")]
    harness.state["slots"] = slots
    sub.answers = {"fresh": json.dumps([_finding("again", "note", summary="Tables are proposed again",
                                                rec="The new accessibility need may justify reconsideration.")])}
    new_reason = "Reconsider tables because an accessible text export is now required."
    changed = {**DECK_SPEC, "decisions": [{"choice": "tables", "rejected": ["charts alone"], "why": new_reason}]}
    assert _control(_call(cold, changed, plan="Add the newly requested accessible export."))["closed"]
    request = sub.calls[-1]["request"]
    if delivery == "packet":
        prepared = _user_text(request.messages[1]["content"])
    elif delivery == "native":
        prepared = NativeToolRoundReviewExecutor(ReviewAssignment(
            request=request, slot=slots[0], call_id="history-preparation")).episode_prompt
    else:
        prepared = request.slot_session_tasks.get("fresh") or request.session_task
    for text in (RATIONALE, OWNER_CHOICES, RAW_REASON, recommendation, "Check caption spelling", new_reason):
        assert text in prepared
    packet_history = json.loads(prepared.split("### Dispute history\n\n```json\n", 1)[1].split("\n```", 1)[0])
    assert packet_history == dispute
    assert "is not reviewer agreement" in prepared
    assert first["aggregate"] == "GREEN"
    after = _history(harness, cold)
    assert any(r["request_fingerprint"] == first["request_fingerprint"] and r["aggregate"] == "GREEN"
               for r in after["dispute_history"]["rounds"])
    assert len(sub.calls) == 3 and _state(harness)["cycles_paid"] == before["cycles_paid"] + 1
    # Exact free replay neither buys another panel nor changes the saved dispute.
    saved = _state(harness)
    _call(cold, changed, plan="Add the newly requested accessible export.")
    assert len(sub.calls) == 3 and _state(harness) == saved


def _wave(root, cycle, previous=None, *, fingerprint=None, **changes):
    spec = plan_spec.normalize_spec({**DECK_SPEC, "goal": "Ship"})[0]
    wave = {"cycle_index": cycle, "request_fingerprint": fingerprint or f"{cycle:064x}",
            "spec": spec, "spec_hash": plan_spec.spec_hash(spec), "aggregate": "REVIEW_REQUIRED",
            "closed": False, "paid": True, "findings": [], "dispositions": [], "actors": [],
            "reviewer_outputs": [], **changes}
    if previous:
        wave.update(previous_fingerprint=previous["request_fingerprint"], previous_wave_artifact=previous["wave_artifact"])
    return {**wave, "wave_artifact": artifacts.persist_wave(root, "task-1", wave)}


def test_evicted_rounds_same_subject_and_superseded_answers_keep_original_verdict(tmp_path):
    first = _wave(tmp_path, 1, fingerprint="a" * 64,
                  findings=[{"finding_id": "s1:f", "class": "blocking", "summary": "original objection"}],
                  aggregate="REVISE_PLAN")
    answered = _wave(tmp_path, 1, fingerprint="a" * 64, supersedes_wave_artifact=first["wave_artifact"],
                     aggregate="GREEN", closed=True,
                     dispositions=[{"finding_id": "s1:f", "decision": "reject", "rationale": RATIONALE}])
    revised = _wave(tmp_path, 1, fingerprint="a" * 64, supersedes_wave_artifact=answered["wave_artifact"],
                    dispositions=[{"finding_id": "s1:f", "decision": "accept", "rationale": "New owner requirement"}])
    last = revised
    for cycle in range(2, 14):
        last = _wave(tmp_path, cycle, last)
    state = {"waves": [artifacts.compact_wave(last)], "waves_omitted": 12}
    original = copy.deepcopy(state)
    history = artifacts.plan_review_dispute_history(tmp_path, "task-1", state)
    assert history["status"] == "complete" and state == original
    assert len(history["rounds"]) == 15
    assert history["rounds"][0]["aggregate"] == "REVISE_PLAN"
    assert history["rounds"][1]["dispositions"][0]["rationale"] == RATIONALE
    assert history["rounds"][2]["dispositions"][0]["rationale"] == "New owner requirement"
    assert all(r["source"]["source_ref"]["sha256"] for r in history["rounds"])


def test_disposition_revisions_inline_each_distinct_material_once(tmp_path):
    from ouroboros.artifacts import read_actor_source_bytes

    spec = plan_spec.normalize_spec({**DECK_SPEC, "goal": "Ship", "invariants": ["SPEC_MATERIAL " * 800]})[0]
    original = _wave(tmp_path, 1, spec=spec, spec_hash=plan_spec.spec_hash(spec),
                     findings=[{"finding_id": "s1:f", "summary": "One recommendation", "class": "note"}],
                     reviewer_outputs=[{"slot_id": "s1", "text": RAW_REASON,
                                        "request_messages": [{"content": "RECURSIVE_PACKET " * 1000}],
                                        "session_task": "RECURSIVE_SESSION " * 1000}])
    latest = original
    for turn in range(15):
        exact = artifacts.read_wave(tmp_path, "task-1", latest["wave_artifact"])
        exact.update(supersedes_wave_artifact=latest["wave_artifact"],
                     dispositions=[{"finding_id": "s1:f", "decision": "reject", "rationale": f"Distinct reason {turn}"}])
        latest = {**exact, "wave_artifact": artifacts.persist_wave(tmp_path, "task-1", exact)}
    history = artifacts.plan_review_dispute_history(tmp_path, "task-1", {"waves": [artifacts.compact_wave(latest)]})
    text = json.dumps(history)
    assert history["status"] == "complete"
    assert text.count("RAW_REASON_END") == 1
    assert text.count("SPEC_MATERIAL") == 800
    assert "RECURSIVE_PACKET" not in text and "RECURSIVE_SESSION" not in text
    assert len(history["decision_rows"]) == 16
    for turn in range(15):
        assert f"Distinct reason {turn}" in text
    # The aliases name verified existing bytes, rather than another hidden copy.
    alias = history["rounds"][-1]["reviewer_outputs"][0]["text_source"]["same_content_as"]
    assert json.loads(read_actor_source_bytes(tmp_path, "task-1", alias["source_ref"]))["reviewer_outputs"][0]["text"] == RAW_REASON
    assert alias["field"] == "reviewer_outputs[0].text"


def test_selected_author_subject_keeps_its_source_and_no_invented_verdict(harness, monkeypatch):
    ctx, sub, first, _raw, _rec = _first_two_rounds(harness, monkeypatch)
    harness.state["enforcement"] = "advisory"
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    critic = _state(harness)["waves"][-1]
    _call(ctx, {**DECK_SPEC, "invariants": ["New owner requirement"]}, plan="New version chosen by author.",
          review_disposition={"review_fingerprint": critic["request_fingerprint"], "author_action": "finish",
                              "author_disposition": {"disposition": "partial", "rationale": "Accept caption advice; preserve the earlier chart choice."}})
    author = _history(harness, ctx)
    selected = author["dispute_history"]["current_author_plan"]
    assert author["current_wave"] is None
    assert selected["review_fingerprint"] == critic["request_fingerprint"]
    assert selected["fingerprint"] != critic["request_fingerprint"]
    assert "aggregate" not in selected and "verdict" not in selected
    assert _resident_reason(author["dispute_history"], selected["author_disposition"]["rationale"]) == "Accept caption advice; preserve the earlier chart choice."
    assert len(sub.calls) == 2


def test_successive_author_selections_survive_cold_new_attempt(harness, monkeypatch):
    from ouroboros.artifacts import read_actor_source_bytes

    ctx, sub, _first, _raw, _rec = _first_two_rounds(harness, monkeypatch)
    harness.state["enforcement"] = "advisory"
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    critic = _state(harness)["waves"][-1]
    reasons = ["Keep charts, with exact rationale. " * 110 + "FIRST_STANCE_END",
               "Owner now asks for both formats; accept the new requirement. SECOND_STANCE_END"]
    for index, reason in enumerate(reasons):
        answer = _call(ctx, {**DECK_SPEC, "invariants": [f"Selected requirement {index}"]},
            plan=f"Selected author version {index}", review_disposition={
                "review_fingerprint": critic["request_fingerprint"], "author_action": "finish",
                "author_disposition": {"disposition": "partial", "rationale": reason}})
        assert "Current author plan saved" in answer
    before = _state(harness)
    ref = before["author_history_head"]
    exact = json.loads(read_actor_source_bytes(harness.drive, ctx.task_id, ref))
    assert exact["author_disposition"]["rationale"] == reasons[1]
    assert exact["previous_author_subject"]["sha256"]
    assert exact["review_wave_artifact"] == critic["wave_artifact"]
    # The author graph itself names its critic; recovery does not require a hot wave.
    evicted = artifacts.plan_review_dispute_history(harness.drive, ctx.task_id, {**before, "waves": [], "waves_omitted": 2})
    assert evicted["status"] == "complete", evicted["gaps"]
    assert critic["request_fingerprint"] in {wave["request_fingerprint"] for wave in evicted["rounds"]}
    cold = harness.make_ctx()
    sub.answers = {"s1": CLEAN, "s2": CLEAN, "s3": CLEAN}
    _call(cold, {**DECK_SPEC, "invariants": ["Final version sent to a critic"]}, plan="New critic version")
    history = _history(harness, cold)["dispute_history"]
    assert history["status"] == "complete", history["gaps"]
    assert [_resident_reason(history, p["author_disposition"]["rationale"]) for p in history["author_selections"]] == reasons
    assert all("verdict" not in p and "aggregate" not in p for p in history["author_selections"])
    prepared = _user_text(sub.calls[-1]["request"].messages[1]["content"])
    assert all(reason in prepared for reason in reasons)
    assert len(sub.calls) == 3 and _state(harness)["cycles_paid"] == before["cycles_paid"] + 1
    assert _state(harness)["author_history_head"] == ref


def test_disposition_rationale_is_retained_exact_before_hot_projection(harness, monkeypatch):
    from ouroboros.task_results import record_plan_review_attempt

    ctx, _sub, _first, _raw, _rec = _first_two_rounds(harness, monkeypatch)
    first = _state(harness)["waves"][-1]
    rationale = "  Exact whitespace\n\n" + "Original argument, alternatives and evidence. " * 350 + "RATIONALE_TAIL\n"
    answer = plan_review._handle_plan_task(ctx, review_disposition={
        "review_fingerprint": first["request_fingerprint"],
        "items": [{"finding_id": "s1:captions", "decision": "reject", "rationale": rationale}]})
    assert "ERROR" not in answer
    record_plan_review_attempt(harness.drive, ctx.task_id, fingerprint="f" * 64)
    history = _history(harness, harness.make_ctx())["dispute_history"]
    assert any(rationale in _resident_reason(history, d["authored_view"])["author_rationales"]
               for wave in history["rounds"] for d in wave.get("dispositions", []))
    assert any(rationale in row["reason"].get("author_rationales", [])
               for row in history["decision_rows"] if isinstance(row["reason"], dict))
    wave = next(w for w in _state(harness)["waves"] if w["request_fingerprint"] == first["request_fingerprint"])
    exact = artifacts.read_wave(harness.drive, ctx.task_id, wave["wave_artifact"])
    assert exact["dispositions"][0]["rationale"] == rationale
    _items, error = plan_review._disposition_items(exact, [{"finding_id": "s1:tables", "rationale": {"bad": "shape"}}])
    assert "rationale must be text" in error


@pytest.mark.parametrize("missing_source", [False, True])
def test_reachable_legacy_author_stance_is_retained_on_next_attempt(harness, monkeypatch, missing_source):
    from ouroboros.artifacts import store_actor_source_bytes
    from ouroboros.review_records import build_author_disposition
    from ouroboros.task_results import _update_plan_review_state, record_plan_review_attempt

    spec = plan_spec.normalize_spec({**DECK_SPEC, "goal": "Ship"})[0]
    fp = "d" * 64
    source = {"kind": "plan_author_subject", "fingerprint": fp, "goal": "Ship", "spec": spec, "plan_prose": "Legacy chosen plan"}
    ref = store_actor_source_bytes(harness.drive, "task-1", category="context_checkpoints",
        source_id="legacy-author", data=json.dumps(source).encode(), extension="json")
    author = build_author_disposition(disposition="partial", rationale="Genuine older stance", subject_hash=fp)
    _update_plan_review_state(harness.drive, "task-1", lambda state: {**state, "current_attempt": {
        "fingerprint": fp, "status": "open", "reason": "author_current_plan",
        "author_subject": {"source_ref": ref, "review_fingerprint": "c" * 64, "author_disposition": author}}})
    if missing_source:
        (task_artifact_dir_path(harness.drive, "task-1", create=False) / ref["path"]).unlink()
        before = artifacts.plan_review_dispute_history(harness.drive, "task-1", _state(harness))
        assert _resident_reason(before, before["author_selections"][0]["author_disposition"]["rationale"]) == "Genuine older stance"
    record_plan_review_attempt(harness.drive, "task-1", fingerprint="e" * 64)
    history = artifacts.plan_review_dispute_history(harness.drive, "task-1", _state(harness))
    [selection] = history["author_selections"]
    assert selection["selected_source_ref"] == ref
    assert _resident_reason(history, selection["author_disposition"]["rationale"]) == "Genuine older stance"
    assert history["current_author_plan"] is None
    assert history["status"] == "source_unavailable"
    assert any("legacy author selection" in gap["reason"] for gap in history["gaps"])


def test_lost_selected_source_keeps_known_stance_and_exact_critic_without_blocking_new_attempt(tmp_path):
    from ouroboros.artifacts import store_actor_source_bytes
    from ouroboros.review_records import build_author_disposition
    from ouroboros.task_results import record_plan_review_attempt

    critic = _wave(tmp_path, 1)
    fp = "b" * 64
    value = {"kind": "plan_author_subject", "fingerprint": fp, "spec": critic["spec"], "plan_prose": "Chosen plan"}
    ref = store_actor_source_bytes(tmp_path, "task-1", category="context_checkpoints", source_id="selected",
                                  data=json.dumps(value).encode(), extension="json")
    author = build_author_disposition(disposition="partial", rationale="Known author reason survives file loss", subject_hash=fp)
    selected = record_plan_review_attempt(tmp_path, "task-1", fingerprint=fp, author_subject={
        "source_ref": ref, "author_disposition": author, "review_fingerprint": critic["request_fingerprint"],
        "review_wave_artifact": critic["wave_artifact"]})
    lost = selected["author_history_head"]
    (task_artifact_dir_path(tmp_path, "task-1", create=False) / lost["path"]).unlink()
    for state in (selected, record_plan_review_attempt(tmp_path, "task-1", fingerprint="c" * 64)):
        history = artifacts.plan_review_dispute_history(tmp_path, "task-1", state)
        assert history["status"] == "source_unavailable"
        selected_author = history["author_selections"][0]["author_disposition"]
        assert {**selected_author, "rationale": _resident_reason(history, selected_author["rationale"])} == author
        assert history["rounds"][0]["request_fingerprint"] == critic["request_fingerprint"]
        assert "spec" not in history["author_selections"][0]  # no invented replacement plan
    assert state["current_attempt"]["fingerprint"] == "c" * 64 and state["cycles_paid"] == 0


def test_late_feedback_stays_historical_and_missing_source_is_a_gap(tmp_path):
    from tests.test_plan_review_historical_supplements import wave, complete, state, TASK
    from ouroboros.tools.plan_review_collect import attach_historical_results

    original, request, slots = wave(tmp_path, closed=True)
    actor, raw = complete(tmp_path, original, request, slots[0])
    assert attach_historical_results(tmp_path, TASK, fingerprint=original["request_fingerprint"]) == 1
    stored = state(tmp_path)
    history = artifacts.plan_review_dispute_history(tmp_path, TASK, stored)
    feedback = history["rounds"][-1]["historical_feedback"][0]
    assert feedback["result"]["text"] == raw
    assert feedback["result"]["operation_id"] == actor.operation_id
    assert history["rounds"][-1]["aggregate"] == original["aggregate"]
    source = feedback["source"]["source_ref"]
    (task_artifact_dir_path(tmp_path, TASK) / source["path"]).unlink()
    gap = artifacts.plan_review_dispute_history(tmp_path, TASK, stored)
    assert gap["status"] == "source_unavailable"
    assert gap["rounds"][-1]["aggregate"] == original["aggregate"]
    assert gap["gaps"][0]["code"] == "PLAN_REVIEW_SOURCE_UNAVAILABLE"
    assert state(tmp_path) == stored


def test_missing_secondary_source_does_not_erase_the_verified_wave(tmp_path):
    from ouroboros.artifacts import store_actor_source_bytes

    ref = store_actor_source_bytes(tmp_path, "task-1", category="context_checkpoints", source_id="room",
                                   data=b"room source", extension="jsonl")
    wave = _wave(tmp_path, 1, dialogue_source_ref=ref,
                 dispositions=[{"finding_id": "s1:f", "decision": "reject", "rationale": RATIONALE}])
    (task_artifact_dir_path(tmp_path, "task-1") / ref["path"]).unlink()
    history = artifacts.plan_review_dispute_history(tmp_path, "task-1", {"waves": [wave]})
    assert history["status"] == "source_unavailable"
    assert history["rounds"][0]["dispositions"][0]["rationale"] == RATIONALE


def test_missing_predecessor_is_a_typed_gap_beside_available_decisions(harness, monkeypatch):
    ctx, sub, first, _raw, _rec = _first_two_rounds(harness, monkeypatch)
    # Damage only a fixture's superseded original; the answered wave still has
    # its rejection and the second wave remains readable.
    path = task_artifact_dir_path(harness.drive, ctx.task_id) / first["wave_artifact"]["path"]
    path.unlink()
    before = _state(harness)
    history = _history(harness, ctx)["dispute_history"]
    assert history["status"] == "source_unavailable"
    assert any(g["code"] == "PLAN_REVIEW_SOURCE_UNAVAILABLE" for g in history["gaps"])
    assert RATIONALE in json.dumps(history)
    # Even a cold, direct packet builder carries the explicit missing-source fact.
    from ouroboros.tools.plan_review_runtime import build_plan_review_packet
    latest = before["waves"][-1]
    _system, packet, session = build_plan_review_packet(
        ctx, spec=latest["spec"], request=SimpleNamespace(plan="Next"), manifest={}, constitutional=False,
        system_root=harness.system, active_root=harness.workspace, cycle_index=3,
        enforcement="blocking", previous=latest, state=before)
    assert "PLAN_REVIEW_SOURCE_UNAVAILABLE" in packet and RATIONALE in session
    assert _state(harness) == before and len(sub.calls) == 2


def test_pending_projection_does_not_collect_or_dispatch(harness, monkeypatch):
    from tests.test_plan_review_reconciliation import _install_barrier_substrate

    calls = []
    _install_barrier_substrate(monkeypatch, calls, still_pending={"s1"})
    ctx = harness.make_ctx()
    _call(ctx)
    before = _state(harness)
    snapshots = [_history(harness, ctx) for _ in range(2)]
    assert snapshots[0] == snapshots[1]
    assert snapshots[0]["current_wave"]["custody_pending"]
    assert any(not row.get("ok") for wave in snapshots[0]["dispute_history"]["rounds"] for row in wave["reviewers"])
    assert _state(harness) == before and len(calls) == 1


def test_unresolvable_index_and_lineage_loop_are_gaps(tmp_path):
    compact = artifacts.compact_wave({"cycle_index": 1, "request_fingerprint": "a" * 64})
    gap = artifacts.plan_review_dispute_history(tmp_path, "task-1", {"waves": [compact]})
    assert gap["status"] == "source_unavailable" and gap["gaps"]
    spec = plan_spec.normalize_spec({**DECK_SPEC, "goal": "Ship"})[0]
    hot = {"cycle_index": 1, "request_fingerprint": "b" * 64, "previous_fingerprint": "b" * 64, "spec": spec}
    loop = artifacts.plan_review_dispute_history(tmp_path, "task-1", {"waves": [hot]})
    assert loop["status"] == "source_unavailable" and len(loop["rounds"]) == 1
    assert "cycle" in loop["gaps"][0]["reason"]
