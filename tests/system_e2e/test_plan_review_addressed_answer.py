"""S34-S35 — plan review's ANSWER CHANNEL on a real isolated server, keyless.

Both scenarios run under BLOCKING enforcement at the shipped cycle cap (2), over the
asynchronous barrier route (a fresh dispatch returns at the dispatch barrier with an
open custody-pending wave; the settled-wave mailbox frame is the cue; the identical
envelope resubmitted after it is the $0 collection), with three DISTINCT keyless
reviewer models so the stub answers PER SEAT by the wire model id.

* S34 — the addressed answer: cycle 1 ends REVIEW_REQUIRED (t1 blocking below
  quorum, t2/t3 clean); the author records a $0 reject; the IDENTICAL envelope sent
  with that answer asks t1 ALONE (one paid cycle, same fingerprint) while t2 and t3
  keep their recorded answers at $0 as replayed rows; t1 retires its finding; the
  collection closes the wave GREEN; two cycles paid; the task completes under
  blocking. The durable chain is exact: the barrier snapshot of cycle 2 names the
  cycle-1 wave it answered, and t1's second request carries the prior cycle's
  rationale.
* S35 — the no-need path: t1 asks the author (need_evidence) and t2 leaves a note;
  a $0 accept of the question closes the wave GREEN with no second panel (three
  reviewer calls in total, one paid cycle) and the task completes under blocking.

Every assertion reads durable artifacts (the stored task row, the immutable wave
artifacts, the tool log) and the stub's own call ledger — never an HTTP 200 alone.
"""

from __future__ import annotations

import json
import re

import pytest

from ouroboros.tools.review_synthesis import PLAN_REVIEW_CONTROL_PREFIX
from tests.system_e2e.harness import (
    DISTINCT_MOCK_MODEL_IDS,
    KEYLESS_REVIEW_ROWS,
    LANE_MOCK,
    ArtifactOracle,
    ReviewScript,
    body_text,
    default_slot_binder,
    keyless_review_catalog,
    keyless_settings,
    require_lane,
    start_server,
    submit_running,
    wait_durable_result,
)
from tests.system_e2e.test_system_scenarios_w3a import (
    _ROOT_TASK_ID_RE,
    _WAIT_ROUNDS_MAX,
    _WAIT_WINDOW_SEC,
    _Again,
    _HoldingStubModel,
    _tool_rows,
)

CLEAN = "[]\nNO_FINDINGS"
# A reviewer's seat id IS its catalog row id: seat t<i> (the ``-t<i>`` tail of the
# wire model id the stub answers by) sits on row ``review-t<i>``, and the durable
# wave names seats and findings (``<row>:<finding id>``) by the row.
R1, R2, R3 = KEYLESS_REVIEW_ROWS
GOAL = "Write the addressed-answer smoke note."
PLAN = "Draft the note with its marker line, verify the marker, then finish."
SPEC = {
    "in_scope": ["addressed-answer smoke note"],
    "acceptance_claims": ["The note carries the S34_MARKER line."],
    # Required on every submitted spec; the smoke note changes no repository file.
    "affected_paths": [],
}
T1_OBJECTION = "The spec has no invariant pinning the exact bytes of the S34_MARKER line."
REJECT_RATIONALE = "claim_1 already pins the exact marker line; an invariant would duplicate it."
T1_BLOCKING = json.dumps([{
    "id": "f1", "class": "blocking", "breaks": "claim_1",
    "summary": T1_OBJECTION,
    "recommendation": "Add an invariant naming the exact marker bytes.",
}])
T1_QUESTION = json.dumps([{
    "id": "q1", "class": "need_evidence", "breaks": "claim_1",
    "summary": "Which exact marker line will the note carry?",
    "recommendation": "",
}])
T2_NOTE = json.dumps([{
    "id": "n1", "class": "note", "breaks": "",
    "summary": "Keep the note to one paragraph.",
    "recommendation": "",
}])

# The settled-wave frame (plan_review_collect.announce_released_settlement) carries
# the fingerprint AND the counts; the addressed cycle re-uses the fingerprint, so the
# frames are told apart by their full text, never by the fingerprint alone.
_FRAME_RE = re.compile(r"Plan review wave ([0-9a-f?]+): (\d+) of (\d+) reviewer slot\(s\) settled")
_FINGERPRINT_RE = re.compile(r"\*\*Plan fingerprint:\*\* `([0-9a-f]{64})`")


def _seat(body: dict) -> str:
    """The seat a plan-review call came from: the ``-t<i>`` tail of the wire model id."""
    return default_slot_binder(body).rsplit("-", 1)[-1]


def _per_seat(answers: dict):
    """A ReviewScript step that answers BY SEAT. An unexpected seat gets a loud
    unparseable text, so the durable wave (not a hang) names the defect."""
    def step(body: dict) -> str:
        return answers.get(_seat(body), f"E2E_SCRIPT_ERROR: unexpected reviewer seat {default_slot_binder(body)!r}")
    return step


def _envelope(**extra) -> dict:
    """The ONE envelope of the scenario; ``extra`` rides beside it (review_disposition)."""
    return {"tool": "plan_task", "arguments": {"goal": GOAL, "plan": PLAN, "spec": SPEC, **extra}}


def _after_frames(count: int, then):
    """Hold the script until ``count`` DISTINCT settled-wave frames are visible in the
    transcript, then emit ``then`` (a step or a callable step). Until then the step
    waits on this task's own id (``wait_task`` returns early on
    ``owner_mailbox_pending``), the route the plan-review contract text names."""
    waits = {"rounds": 0}

    def step(body: dict) -> dict:
        text = body_text(body)
        frames = list(dict.fromkeys(_FRAME_RE.findall(text)))
        if len(frames) >= count:
            return then(body) if callable(then) else then
        waits["rounds"] += 1
        task_ids = _ROOT_TASK_ID_RE.findall(text)
        if not task_ids or waits["rounds"] > _WAIT_ROUNDS_MAX:
            return {"final": (
                f"E2E_SCRIPT_ERROR: fewer than {count} settled-wave frame(s) after "
                f"{waits['rounds']} round(s); frames seen: {frames}; own task id visible: {bool(task_ids)}")}
        return _Again({"tool": "wait_task", "arguments": {
            "task_id": task_ids[-1], "timeout_sec": _WAIT_WINDOW_SEC}})

    return step


def _answer_step(items: list, *, with_envelope: bool):
    """Answer the recorded wave named by the LAST ``Plan fingerprint:`` line the host
    printed: alone (a $0 disposition) or beside the identical envelope (the addressed
    re-ask)."""
    def step(body: dict) -> dict:
        found = _FINGERPRINT_RE.findall(body_text(body))
        if not found:
            return {"final": "E2E_SCRIPT_ERROR: no plan fingerprint visible in the transcript"}
        disposition = {"review_fingerprint": found[-1], "items": list(items)}
        if with_envelope:
            return _envelope(review_disposition=disposition)
        return {"tool": "plan_task", "arguments": {"review_disposition": disposition}}

    return step


def _control(text: str) -> dict:
    lines = [line for line in str(text).splitlines() if line.startswith(PLAN_REVIEW_CONTROL_PREFIX)]
    assert len(lines) == 1, text[-600:]
    return json.loads(lines[0][len(PLAN_REVIEW_CONTROL_PREFIX):])


def _settings(stub) -> dict:
    return keyless_settings(
        stub, OUROBOROS_RUNTIME_MODE="advanced", OUROBOROS_REVIEW_ENFORCEMENT="blocking",
        OUROBOROS_SUBAGENTS=keyless_review_catalog(distinct_models=True),
    )


def _wave_artifacts(oracle: ArtifactOracle) -> list:
    paths = sorted((oracle.data_root / "task_results" / "artifacts").rglob("plan-review-wave-*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _plan_results(oracle: ArtifactOracle, task_id: str) -> list:
    """The FULL text of every ``plan_task`` result of the task, in call order, read
    through each tool row's persisted call trace (the direct tools.jsonl row keeps a
    bounded preview; the exact result is the digest-verified observability blob)."""
    from ouroboros.observability import read_call_payload

    task_drive = oracle.task_drive(task_id)
    out = []
    for row in _tool_rows(task_drive, "plan_task"):
        call_id = str((row.get("result_ref") or {}).get("call_id") or "")
        assert call_id, row
        _manifest, payload, _ref = read_call_payload(task_drive.data_root, task_id=task_id, call_id=call_id)
        out.append(str(payload.get("result") or ""))
    return out


# ===========================================================================
# S34 — the addressed answer under blocking, barrier route
# ===========================================================================

@pytest.mark.integration
@pytest.mark.serial
def test_s34_addressed_answer_reasks_the_objector_alone_and_closes_green(e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    from ouroboros.tools.plan_review_artifacts import read_wave

    root = tmp_path_factory.mktemp("s34")
    reject = {"finding_id": f"{R1}:f1", "decision": "reject", "rationale": REJECT_RATIONALE}
    review_script = ReviewScript({"plan_review": [
        *([_per_seat({"t1": T1_BLOCKING, "t2": CLEAN, "t3": CLEAN})] * 3),  # cycle 1: three seats
        _per_seat({"t1": CLEAN}),                                            # cycle 2: t1 alone
    ]})
    stub = _HoldingStubModel(
        [_envelope(),                                                   # dispatch: the barrier
         _after_frames(1, _envelope()),                                 # $0 collect: REVIEW_REQUIRED
         _answer_step([reject], with_envelope=False),                   # $0 reject, wave stays open
         _answer_step([reject], with_envelope=True),                    # addressed re-ask: t1 alone
         _after_frames(2, _envelope())],                                # $0 collect: GREEN closed
        review_script=review_script, model_ids=DISTINCT_MOCK_MODEL_IDS,
    )
    with stub:
        server = start_server(e2e_clone, root, _settings(stub))
        try:
            task_id = submit_running(
                server, "Plan the note through plan_task, answer the reviewer, then finish.")
            result = server.wait_task(task_id, timeout=600)
            assert result.get("status") == "completed", result
            oracle = ArtifactOracle(server.data_root)
            stored = wait_durable_result(oracle, task_id)
            assert stored.get("status") == "completed", stored.get("status")

            # The durable chronicle: two PAID cycles on ONE fingerprint, GREEN and closed.
            state = stored.get("plan_review_state")
            assert isinstance(state, dict), sorted(stored)
            assert int(state.get("cycles_paid") or 0) == 2, state
            waves = [w for w in (state.get("waves") or []) if isinstance(w, dict)]
            last = waves[-1]
            fingerprint = str(last.get("request_fingerprint") or "")
            assert len(fingerprint) == 64 and all(w.get("request_fingerprint") == fingerprint for w in waves), waves
            assert last.get("aggregate") == "GREEN" and last.get("closed") is True, last
            assert int(last.get("cycle_index") or 0) == 2 and last.get("paid") is True, last
            assert last.get("findings") == [] and last.get("dispositions") == [], last
            actors = {str(a.get("slot_id")): a for a in last.get("actors") or []}
            assert sorted(actors) == sorted(KEYLESS_REVIEW_ROWS), actors
            for sid in (R2, R3):  # kept at $0: the replayed cycle-1 rows, never a send
                kept = actors[sid]
                assert kept.get("operation_state") == "not_dispatched" and kept.get("cost") == 0.0, kept
                assert kept.get("replayed_from", {}).get("cycle_index") == 1, kept
                assert kept["replayed_from"].get("request_fingerprint") == fingerprint, kept
            assert "replayed_from" not in actors[R1] and actors[R1].get("ok") is True, actors[R1]
            assert str(actors[R1].get("model") or "").endswith("mock-model-t1"), actors[R1]

            # The stub saw 3 + 1 plan-review calls: three distinct seats, then t1 alone,
            # whose second request carries the prior cycle's rationale.
            plan_bodies = [body for kind, body in stub.calls if kind == "plan_review"]
            assert len(plan_bodies) == 4, stub.kinds()
            assert sorted(default_slot_binder(b) for b in plan_bodies[:3]) == sorted(DISTINCT_MOCK_MODEL_IDS)
            assert default_slot_binder(plan_bodies[3]) == "mock-model-t1", plan_bodies[3].get("model")
            second_request = body_text(plan_bodies[3])
            assert "PRIOR CYCLES" in second_request and REJECT_RATIONALE in second_request, second_request[-3000:]
            assert T1_OBJECTION in second_request, "t1's own cycle-1 answer continues its transcript"

            # The exact artifact chain: the cycle-2 BARRIER snapshot names the seats it asked
            # again and the cycle-1 wave it answered; that reference reads back as the
            # answered wave (cycle 1, same fingerprint, the reject recorded); the collected
            # cycle-2 wave carries the same predecessor reference AND the same addressed
            # lineage (the collection re-records the cycle it settles) and is closed GREEN.
            barrier = [p for p in _wave_artifacts(oracle)
                       if p.get("custody_pending") and int(p.get("cycle_index") or 0) == 2]
            assert len(barrier) == 1, [(p.get("cycle_index"), p.get("custody_pending")) for p in _wave_artifacts(oracle)]
            addressed = barrier[0].get("addressed")
            assert addressed and addressed["slots"] == [R1] and addressed["finding_ids"] == [f"{R1}:f1"], addressed
            assert addressed["kept"] == [R2, R3], addressed
            answered = read_wave(oracle.data_root, task_id, addressed["wave_artifact"])
            assert int(answered.get("cycle_index") or 0) == 1 and answered.get("request_fingerprint") == fingerprint
            assert answered.get("closed") is False and answered.get("aggregate") == "REVIEW_REQUIRED", answered.get("aggregate")
            assert [(d["finding_id"], d["decision"]) for d in answered.get("dispositions") or []] == [(f"{R1}:f1", "reject")]
            assert last.get("previous_wave_artifact") == addressed["wave_artifact"], last.get("previous_wave_artifact")
            collected = read_wave(oracle.data_root, task_id, last["wave_artifact"])
            assert collected.get("closed") is True and collected.get("aggregate") == "GREEN"
            assert int(collected.get("cycle_index") or 0) == 2
            assert last.get("addressed") == addressed and collected.get("addressed") == addressed, last.get("addressed")

            # FIVE plan_task calls bought exactly one re-ask: dispatch, collect, the $0
            # reject, the addressed re-ask (barrier), collect.
            results = _plan_results(oracle, task_id)
            assert len(results) == 5, [text[:120] for text in results]
            assert [(_control(t)["outcome"], _control(t)["closed"]) for t in results] == [
                ("DEGRADED", False), ("REVIEW_REQUIRED", False), ("REVIEW_REQUIRED", False),
                ("DEGRADED", False), ("GREEN", True)], results
            assert "REVIEW CUSTODY PENDING" in results[0] and "REVIEW CUSTODY PENDING" in results[3]
            assert (f"**Addressed answer:** asked again {R1} on {R1}:f1; kept at $0: {R2}, {R3}."
                    in results[3]), results[3]
            assert "kept its cycle-1 answer at $0" in results[4], results[4]
            review_script.assert_consumed()
            assert stub.script_consumed()
        finally:
            server.stop()


# ===========================================================================
# S35 — the no-need path under blocking
# ===========================================================================

@pytest.mark.integration
@pytest.mark.serial
def test_s35_no_need_path_closes_at_zero_cost_under_blocking(e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s35")
    accept = {"finding_id": f"{R1}:q1", "decision": "accept",
              "rationale": "The note carries the literal line S34_MARKER as its first line."}
    review_script = ReviewScript({"plan_review": [
        _per_seat({"t1": T1_QUESTION, "t2": T2_NOTE, "t3": CLEAN})] * 3})
    stub = _HoldingStubModel(
        [_envelope(),                                                   # dispatch: the barrier
         _after_frames(1, _envelope()),                                 # $0 collect: REVIEW_REQUIRED
         _answer_step([accept], with_envelope=False)],                  # $0 accept: GREEN closed
        review_script=review_script, model_ids=DISTINCT_MOCK_MODEL_IDS,
    )
    with stub:
        server = start_server(e2e_clone, root, _settings(stub))
        try:
            task_id = submit_running(
                server, "Plan the note through plan_task, answer the reviewer's question, then finish.")
            result = server.wait_task(task_id, timeout=600)
            assert result.get("status") == "completed", result
            oracle = ArtifactOracle(server.data_root)
            stored = wait_durable_result(oracle, task_id)
            assert stored.get("status") == "completed", stored.get("status")

            state = stored.get("plan_review_state")
            assert isinstance(state, dict), sorted(stored)
            assert int(state.get("cycles_paid") or 0) == 1, state
            waves = [w for w in (state.get("waves") or []) if isinstance(w, dict)]
            last = waves[-1]
            assert last.get("aggregate") == "GREEN" and last.get("closed") is True, last
            assert int(last.get("cycle_index") or 0) == 1 and "addressed" not in last, last
            assert [(f["finding_id"], f["class"]) for f in last.get("findings") or []] == [
                (f"{R1}:q1", "need_evidence"), (f"{R2}:n1", "note")], last.get("findings")
            assert [(d["finding_id"], d["decision"]) for d in last.get("dispositions") or []] == [(f"{R1}:q1", "accept")]
            assert state.get("need_evidence_seen") == [], state.get("need_evidence_seen")  # no locator asked
            actors = {str(a.get("slot_id")): a for a in last.get("actors") or []}
            assert sorted(actors) == sorted(KEYLESS_REVIEW_ROWS) and not any("replayed_from" in a for a in actors.values())

            # Exactly one panel of three distinct seats; the accept sent nothing.
            plan_bodies = [body for kind, body in stub.calls if kind == "plan_review"]
            assert len(plan_bodies) == 3, stub.kinds()
            assert sorted(default_slot_binder(b) for b in plan_bodies) == sorted(DISTINCT_MOCK_MODEL_IDS)
            results = _plan_results(oracle, task_id)
            assert len(results) == 3, [text[:120] for text in results]
            assert [(_control(t)["outcome"], _control(t)["closed"]) for t in results] == [
                ("DEGRADED", False), ("REVIEW_REQUIRED", False), ("GREEN", True)], results
            assert "(cached exact review — no reviewer was called)" not in results[1], results[1]
            review_script.assert_consumed()
            assert stub.script_consumed()
        finally:
            server.stop()
