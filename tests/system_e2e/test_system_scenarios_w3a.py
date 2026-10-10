"""S14-S17 — Ф4 wave 3a of the deep-integration suite (v7next plan §8).

Review surfaces and acceptance, keyless throughout, on the wave-1/2 skeleton.
Every scenario asserts DURABLE artifacts (never an HTTP 200 alone, never a
harness exit code) and synchronizes by durable-event polling:

* S14 — PLAN REVIEW: a scripted task drives ``plan_task`` through a real
  REVISE→ACCEPT cycle over the ASYNCHRONOUS event route, which is the whole
  point of the scenario — a fresh dispatch returns at the dispatch barrier with
  an OPEN custody-pending wave (no verdict yet), the released reviewer slots
  settle under process-local custody, the LAST settlement writes ONE system
  frame into this task's mailbox, and the IDENTICAL envelope resubmitted after
  that frame is the $0 collection that closes the wave and pays the cycle
  exactly once. So the scripted agent dispatches, waits on its own task id
  (the wait returns on ``owner_mailbox_pending``), collects, and only then
  revises (cycle 1: every slot returns a blocking finding → REVISE_PLAN;
  cycle 2: a CHANGED spec → all-clean → GREEN closed). The durable chronicle is
  honest (``plan_review_state`` on the stored task row: paid-cycle count, wave
  aggregates; immutable per-wave artifacts with the exact reviewer outputs),
  and the shared owner cycle cap (``OUROBOROS_REVIEW_MAX_CYCLES``) is
  respected: a third paid cycle is refused with the typed
  ``PLAN_REVIEW_CYCLES_EXHAUSTED`` result at $0 (no reviewer dispatched) plus
  the durable ``review_cycles_exhausted`` escalation event.
* S15 — COMMIT TRIAD+SCOPE, ADVISORY: critical feedback returns before Git
  effects; the scripted author reads its real reference and explicitly continues
  without another paid panel. The commit lands with a hash-bound author record,
  original criticism and the loud ``review_advisory_override`` event/counter.
  S15+S16 assert the same organ under both enforcement values.
* S16 — COMMIT TRIAD+SCOPE, BLOCKING enforcement class: a critical triad FAIL
  blocks the commit (repo HEAD does not move), a byte-identical resubmission is
  refused FREE with the typed ``IDENTICAL_DIFF_REFUSED`` (no reviewer paid
  twice for the same bytes), and a fixed diff passes clean review and lands.
  Plus the post-verdict revalidation contract, pinned live: the staged material
  is mutated WHILE the paid triad+scope wave is in flight (after the
  pre-dispatch fingerprint, before settlement) — verdicts come back all-clean
  and the commit is STILL refused (``REVIEW_REVALIDATION_FAILED``, block_reason
  ``revalidation_failed``, fingerprint_status ``mismatch``): a verdict for
  other bytes is never carried forward. (The advisory-freshness contract
  retired with the advisory pipeline, decision 3A: a preflight is one named
  ``review_change(surface=preflight)`` look, never a commit precondition.)
* S17 — ACCEPTANCE LOOP (required + blocking): the terminal runs the real
  acceptance dialogue — panel 1 rejects with an actionable capsule, the loop
  feeds the improvement note back, the agent reworks, panel 2 accepts clean
  (``accepted``/clean pass, both paid identities on the durable wallet). The
  folded A-material invariants hold live: a REWORK that changes nothing is a
  FREE replay — the identical paid identity is refused without buying a third
  panel (``finalized_unaccepted`` / identical refusal, acceptance stub calls
  unchanged) — the keyless instance of the $0-refusal class.

Covered by the other waves (manifest in ``tests/system_e2e/harness.py``):
delegated transport S11-S12, skills lifecycle S13 (wave 3b), self-evolution
absorb and the update variations in wave 4. Still deferred: gateway/UI truth
(Playwright).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess

import pytest

from tests.system_e2e.harness import (
    KEYLESS_PACKET_ROWS,
    KEYLESS_REVIEW_ROWS,
    LANE_MOCK,
    NATIVE_EPISODE_MARKER,
    PLAN_REVIEW_MARKER,
    REVIEW_KINDS,
    ArtifactOracle,
    ReviewScript,
    ScriptedStubModel,
    body_text,
    classify_call,
    clone_repo,
    keyless_settings,
    require_lane,
    two_part_clean_text,
    scripted_completion,
    start_server,
    submit_running,
    wait_durable_result,
)

# ===========================================================================
# Default lane: pins for the wave-3a harness surface (no server, no sockets).
# ===========================================================================


def _plan_body() -> dict:
    return {"messages": [
        {"role": "system", "content": PLAN_REVIEW_MARKER + "\n... rubric ..."},
        {"role": "user", "content": "Plan packet [FINALIZE_NOW] quoted from a transcript"},
    ], "model": "mock-model"}


def _advisory_episode_body(surface: str = "advisory_review") -> dict:
    return {"messages": [
        {"role": "system", "content": "NATIVE REVIEW INSTRUCTIONS"},
        {"role": "user", "content": (
            f"{NATIVE_EPISODE_MARKER} read-only native inspection episode.\n"
            f"Surface: {surface}\nRole hint: advisory pre-reviewer\n\n[OWNER_STOP] quoted"
        )},
    ], "tools": [{"type": "function", "function": {"name": "read_file"}}],
        "model": "mock-model"}


def test_w3a_classification_plan_and_native_episode_branches():
    """The two wave-3a review branches classify BEFORE finalization (roast F22)
    and by name; an unknown native surface stays a typed native_episode."""
    assert classify_call(_plan_body()) == "plan_review"
    assert classify_call(_advisory_episode_body()) == "advisory_review"
    assert classify_call(_advisory_episode_body(surface="something_else")) == "native_episode"
    assert {"plan_review", "advisory_review", "native_episode"} <= REVIEW_KINDS


def test_w3a_canned_plan_and_advisory_answers_parse_under_the_trees_own_parsers():
    """The canned clean answers must be verified-clean under the REAL parsers:
    plan_spec.parse_findings for the plan packet, the advisory clean predicate
    (shared empty_array_is_verified_clean) for the native episode."""
    from ouroboros.tools.plan_spec import parse_findings
    from ouroboros.triad_review import empty_array_is_verified_clean

    _kind, plan_msg = scripted_completion(_plan_body(), 1, lambda _b: None, "x")
    assert _kind == "plan_review"
    findings, parse_error = parse_findings(plan_msg["content"])
    assert findings == [] and parse_error is None

    _kind, adv_msg = scripted_completion(_advisory_episode_body(), 1, lambda _b: None, "x")
    assert _kind == "advisory_review"
    assert empty_array_is_verified_clean(adv_msg["content"])


def test_w3a_review_script_consumes_in_order_then_falls_back_to_canned():
    script = ReviewScript({
        "plan_review": ["RED-1", lambda body: "RED-2:" + body.get("model", "")],
        "triad_review": [{"role": "assistant", "content": "TRIAD-RED"}],
    })
    kind, msg = scripted_completion(_plan_body(), 1, lambda _b: None, "x", review_next=script)
    assert (kind, msg["content"]) == ("plan_review", "RED-1")
    kind, msg = scripted_completion(_plan_body(), 2, lambda _b: None, "x", review_next=script)
    assert (kind, msg["content"]) == ("plan_review", "RED-2:mock-model")
    # Queue exhausted -> canned clean, and assert_consumed is now green.
    kind, msg = scripted_completion(_plan_body(), 3, lambda _b: None, "x", review_next=script)
    assert kind == "plan_review" and msg["content"].startswith("[]")
    with pytest.raises(AssertionError, match="never served"):
        script.assert_consumed()
    triad_body = {"messages": [{"role": "user", "content":
                                "Review the staged diff and context provided in the instructions above."}]}
    kind, msg = scripted_completion(triad_body, 4, lambda _b: None, "x", review_next=script)
    assert (kind, msg["content"]) == ("triad_review", "TRIAD-RED")
    script.assert_consumed()
    assert [k for k, _m in script.served] == ["plan_review", "plan_review", "triad_review"]


def test_w3a_review_script_never_touches_agent_script_steps():
    """A scripted review verdict must not consume agent steps and vice versa —
    the review-organ branch still runs first and owns its own queue."""
    steps = iter([{"tool": "list_files", "arguments": {"path": "."}}])

    def _next(_body):
        return next(steps, None)

    script = ReviewScript({"triad_review": ["TRIAD-RED"]})
    agent_body = {"messages": [{"role": "user", "content": "go"}],
                  "tools": [{"type": "function", "function": {"name": "list_files"}}]}
    kind, msg = scripted_completion(agent_body, 1, _next, "done", review_next=script)
    assert kind == "agent" and msg["tool_calls"][0]["function"]["name"] == "list_files"
    assert not script.consumed()
    with pytest.raises(ValueError, match="review-organ kinds"):
        ReviewScript({"agent": ["nope"]})


# ===========================================================================
# Shared scripted verdicts of the mock-lane scenarios
# ===========================================================================

W3A_TRIAD_RED = json.dumps([{
    "item": "bug_hunting",
    "verdict": "FAIL",
    "severity": "critical",
    "reason": ("scripted critical finding (system_e2e w3a): the smoke note omits its "
               "verification marker line; add the marker before committing."),
}])

W3A_PLAN_RED = json.dumps([{
    "id": "f1",
    "class": "blocking",
    "breaks": "goal",
    "summary": "The spec has no claim binding the note content to a checkable marker.",
    "recommendation": "Add an invariant naming the exact marker the note must carry.",
}])

W3A_ACCEPT_REJECT = json.dumps({
    "verdict": "FAIL",
    "outcome_tier": "best_effort",
    "completion_coach": "Resubmit the final answer with the W3A_DONE marker line included.",
    "dialogue_status": "continue_actionable",
    "criteria_used": [{"criterion": "final answer carries the W3A_DONE marker",
                       "status": "missing", "evidence_refs": []}],
    "findings": [{"severity": "critical", "item": "missing_marker",
                  "evidence": "the candidate final answer does not contain the W3A_DONE marker line",
                  "recommendation": "Add the W3A_DONE marker line to the final answer and resubmit.",
                  "disposition_kind": "new"}],
    "summary": "stub acceptance reviewer: scripted reject round.",
})

# The clean accept cites a section that is ALWAYS host-attested in the packet
# (review_evidence.build_task_acceptance_evidence writes verification_summary
# unconditionally), so the supported criterion's evidence_ref resolves by the
# exact-match vocabulary and the clean bar is reachable keyless.
W3A_ACCEPT_PASS = json.dumps({
    "verdict": "PASS",
    "outcome_tier": "solved",
    "completion_coach": "Nothing further; the marker is present.",
    "dialogue_status": "continue_actionable",
    "criteria_used": [{"criterion": "final answer carries the W3A_DONE marker",
                       "status": "supported", "evidence_refs": ["verification_summary"]}],
    "findings": [],
    "summary": "stub acceptance reviewer: scripted clean accept.",
})


def _tool_rows(oracle: ArtifactOracle, tool_name: str) -> list:  # result rows only: not a call's start / wait end
    return [row for row in oracle.tools_rows() if str(row.get("type") or "tool_call") == "tool_call"
            and str(row.get("tool") or row.get("name") or "") == tool_name]


def _git_log_subjects(clone) -> str:
    return subprocess.run(["git", "log", "-n", "8", "--format=%s"], cwd=str(clone),
                          check=True, capture_output=True, text=True).stdout


def _head(clone) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(clone),
                          check=True, capture_output=True, text=True).stdout.strip()


# ===========================================================================
# S14 — plan review: REVISE→ACCEPT cycle, honest chronicle, cycle cap
# ===========================================================================

S11_GOAL = "Write the w3a plan-review smoke note."
S11_SPEC_V1 = {
    "in_scope": ["w3a plan-review smoke"],
    "acceptance_claims": ["The plan-review smoke completes with a recorded chronicle."],
    # Required on every submitted spec (owner 9=A); the smoke note changes no repository file.
    "affected_paths": [],
}
S11_SPEC_V2 = {
    **S11_SPEC_V1,
    "invariants": ["The note carries the W3A_PLAN marker line (addresses reviewer finding f1)."],
}
S11_SPEC_V3 = {
    **S11_SPEC_V2,
    "non_goals": ["No third paid cycle: this envelope must be refused by the cap."],
}


def _plan_step(spec: dict, note: str) -> dict:
    return {"tool": "plan_task", "arguments": {
        "goal": S11_GOAL,
        "plan": f"Draft the note, verify its content, then finish. ({note})",
        "spec": spec,
    }}


# The ONE system frame plan_review_collect.announce_released_settlement writes
# into the task's mailbox when the last released slot of a wave settles, and the
# task identity the host states in every prompt (the scripted agent waits on its
# OWN id, exactly as the plan-review contract text tells the model to).
_SETTLED_FRAME_RE = re.compile(
    r"Plan review wave ([0-9a-f?]+): \d+ of \d+ reviewer slot\(s\) settled")
_ROOT_TASK_ID_RE = re.compile(r'"root_task_id":\s*"([0-9a-f]{8,})"')
# Bounded so a wave that never settles fails LOUDLY inside the scenario
# (the E2E_SCRIPT_ERROR final answer below) instead of running the task out
# of the harness's own 600s wait with no explanation.
_WAIT_WINDOW_SEC = 60
_WAIT_ROUNDS_MAX = 4


class _Again:
    """Marker a callable step returns to stay on the script one more round."""

    __slots__ = ("step",)

    def __init__(self, step: dict) -> None:
        self.step = step


class _HoldingStubModel(ScriptedStubModel):
    """``ScriptedStubModel`` whose callable steps may HOLD the script index.

    The asynchronous plan-review route needs a step that repeats until a
    precondition is visible in the transcript: the agent waits for the settled-
    wave mailbox frame and only then resubmits the identical envelope. A blind
    sleep inside the step is not an option — ``_answer`` runs under the model's
    call lock, so sleeping there would also block the reviewer-slot calls whose
    settlement it is waiting for. A callable step therefore returns ``_Again``
    to be served now AND decide again on the next round; everything else is
    served and consumed exactly as the base class does, so ``script_consumed()``
    still means "every phase of the scenario ran".
    """

    def _next_step(self, body) -> dict | None:
        index = self._script_index
        step = super()._next_step(body)
        if callable(step):
            step = step(body)
            if isinstance(step, _Again):
                self._script_index = index
                step = step.step
        return step


def _after_wave_settled(wave_ordinal: int, then: dict):
    """Hold the script until the host says plan-review wave ``wave_ordinal``
    settled, then emit ``then``.

    Until the frame is visible the step waits on this task's own id
    (``wait_task`` returns early on ``owner_mailbox_pending``) — the route the
    plan-review contract text names. Acting on an in-flight wave instead would
    either re-read DEGRADED (a collection) or be refused as
    ``PLAN_REVIEW_IN_FLIGHT`` (a revision), so the wait is what makes either
    next move deterministic rather than a race with the reviewer slots.
    """
    waits = {"rounds": 0}

    def step(body: dict) -> dict:
        text = body_text(body)
        settled = list(dict.fromkeys(_SETTLED_FRAME_RE.findall(text)))
        if len(settled) >= wave_ordinal:
            return then
        waits["rounds"] += 1
        task_ids = _ROOT_TASK_ID_RE.findall(text)
        if not task_ids or waits["rounds"] > _WAIT_ROUNDS_MAX:
            return {"final": (
                "E2E_SCRIPT_ERROR: no settled-wave frame for plan-review wave "
                f"{wave_ordinal} after {waits['rounds']} round(s); "
                f"frames seen: {settled}; own task id visible: {bool(task_ids)}")}
        return _Again({"tool": "wait_task", "arguments": {
            "task_id": task_ids[-1], "timeout_sec": _WAIT_WINDOW_SEC}})

    return step


def _collect_step(spec: dict, note: str, *, wave_ordinal: int):
    """The $0 collection of wave ``wave_ordinal``: the IDENTICAL envelope,
    resubmitted after the wave settled. It closes the recorded wave, aggregates
    the verdicts and pays the cycle once, dispatching no second panel."""
    return _after_wave_settled(wave_ordinal, _plan_step(spec, note))


@pytest.mark.integration
@pytest.mark.serial
def test_s14_plan_review_revise_then_accept_cycle_with_honest_chronicle(
        e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s11")
    review_script = ReviewScript({"plan_review": [W3A_PLAN_RED] * 3})
    # Dispatch -> wait for the settled-wave frame -> collect ($0), twice: the
    # honest shape of the asynchronous event route (a fresh call returns at the
    # dispatch barrier, the identical resubmission is the collection).
    stub = _HoldingStubModel(
        [_plan_step(S11_SPEC_V1, "cycle 1"),
         _collect_step(S11_SPEC_V1, "cycle 1", wave_ordinal=1),
         _plan_step(S11_SPEC_V2, "cycle 2 — revised"),
         _collect_step(S11_SPEC_V2, "cycle 2 — revised", wave_ordinal=2)],
        review_script=review_script,
    )
    with stub:
        settings = keyless_settings(stub, OUROBOROS_RUNTIME_MODE="advanced")
        server = start_server(e2e_clone, root, settings)
        try:
            task_id = submit_running(
                server, "Plan the smoke note through plan_task, revise once, then finish.")
            result = server.wait_task(task_id, timeout=600)
            assert result.get("status") == "completed", result
            oracle = ArtifactOracle(server.data_root)
            stored = wait_durable_result(oracle, task_id)

            # Honest durable chronicle on the stored row: two PAID cycles, the
            # current wave GREEN and closed.
            state = stored.get("plan_review_state")
            assert isinstance(state, dict), sorted(stored)
            assert int(state.get("cycles_paid") or 0) == 2, state
            waves = [w for w in (state.get("waves") or []) if isinstance(w, dict)]
            assert waves, state
            assert waves[-1].get("aggregate") == "GREEN", waves[-1]
            assert waves[-1].get("closed") is True, waves[-1]

            # The immutable per-wave artifacts carry the exact reviewer wave
            # bytes. The asynchronous route snapshots each cycle TWICE and both
            # snapshots are evidence: the OPEN barrier wave recorded at dispatch
            # (custody pending, possibly already dispatched) and the wave the $0
            # collection closed. The verdict chronicle is the collected pair:
            # one REVISE_PLAN wave, one GREEN wave. The task artifact store
            # lives under the SERVER data root (task_results/artifacts/), not
            # the task's forked drive; filenames end in a content hash, so the
            # snapshots are keyed by their own facts, never by sort order.
            artifacts = sorted(
                (oracle.data_root / "task_results" / "artifacts").rglob("plan-review-wave-*.json"))
            payloads = [json.loads(path.read_text(encoding="utf-8")) for path in artifacts]
            assert len(payloads) == 4, artifacts
            def by_cycle(payload: dict) -> int:
                return int(payload.get("cycle_index") or 0)

            barrier = sorted((p for p in payloads if p.get("custody_pending")), key=by_cycle)
            from ouroboros.tools.plan_review_artifacts import read_wave

            collected = [read_wave(oracle.data_root, task_id, wave["wave_artifact"])
                         for wave in waves]
            # The durable index names the exact verdict snapshots; the reader
            # verifies their bytes/hash instead of trusting a directory count.
            assert collected == sorted(
                (p for p in payloads if not p.get("custody_pending")), key=by_cycle)
            assert [p.get("aggregate") for p in barrier] == ["DEGRADED"] * 2, barrier
            for pending in barrier:
                states = [actor.get("operation_state") for actor in pending["actors"]]
                assert len(states) == 3
                assert set(states) <= {"pending_dispatch", "in_flight", "settled"}, states
                # One dispatched sibling makes this a paid wave even while
                # another sibling is still waiting at the dispatch barrier.
                assert pending["paid"] is any(state != "pending_dispatch" for state in states)
            assert [p.get("cycle_index") for p in collected] == [1, 2], collected
            assert [p.get("paid") for p in collected] == [True] * 2, collected
            aggregates = [str(p.get("aggregate") or "") for p in collected]
            assert aggregates == ["REVISE_PLAN", "GREEN"], aggregates

            # The REVISE wave chronicled the scripted finding honestly.
            revise_blob = json.dumps(collected[0])
            assert "checkable marker" in revise_blob, revise_blob[:2000]

            # Exactly two paid waves of three slots each hit the model; the
            # scripted red round was fully served. FOUR plan_task calls bought
            # only those two panels: each cycle is one dispatch plus one $0
            # collection of the same envelope, never a second panel.
            assert stub.kinds().count("plan_review") == 6, stub.kinds()
            plan_rows = _tool_rows(oracle.task_drive(task_id), "plan_task")
            assert len(plan_rows) == 4, plan_rows
            review_script.assert_consumed()
            assert stub.script_consumed()
        finally:
            server.stop()


@pytest.mark.integration
@pytest.mark.serial
def test_s14_plan_review_cycle_cap_refuses_third_paid_cycle(e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    from ouroboros.outcomes import REASON_REVIEW_CYCLES_EXHAUSTED

    root = tmp_path_factory.mktemp("s11cap")
    review_script = ReviewScript({"plan_review": [W3A_PLAN_RED] * 6})
    # The cap is enforced at DISPATCH, so this scenario never collects — but it
    # still waits for each panel to settle before submitting the next envelope.
    # A revised envelope sent while the previous panel is still in flight is
    # refused as PLAN_REVIEW_IN_FLIGHT, which would mask the cap refusal under
    # a different typed reason.
    stub = _HoldingStubModel(
        [_plan_step(S11_SPEC_V1, "cycle 1"),
         _after_wave_settled(1, _plan_step(S11_SPEC_V2, "cycle 2 — still open")),
         _after_wave_settled(2, _plan_step(S11_SPEC_V3, "cycle 3 — must be refused by the cap"))],
        review_script=review_script,
    )
    with stub:
        settings = keyless_settings(stub, OUROBOROS_RUNTIME_MODE="advanced")
        server = start_server(e2e_clone, root, settings)
        try:
            task_id = submit_running(
                server, "Plan the note; keep revising until told to stop, then finish.")
            result = server.wait_task(task_id, timeout=600)
            assert result.get("status") == "completed", result
            oracle = ArtifactOracle(server.data_root)
            stored = wait_durable_result(oracle, task_id)

            # The shared owner cap (OUROBOROS_REVIEW_MAX_CYCLES default) bounded
            # the organ: exactly TWO paid waves, the third call refused at $0.
            assert stub.kinds().count("plan_review") == 6, stub.kinds()
            state = stored.get("plan_review_state")
            assert isinstance(state, dict), sorted(stored)
            assert int(state.get("cycles_paid") or 0) == 2, state
            current = state.get("current_attempt")
            assert isinstance(current, dict) and current.get("status") == "cycles_exhausted", state

            # The third plan_task tool result is the typed refusal.
            task_drive = oracle.task_drive(task_id)
            plan_rows = _tool_rows(task_drive, "plan_task")
            assert len(plan_rows) == 3, plan_rows
            assert "PLAN_REVIEW_CYCLES_EXHAUSTED" in json.dumps(plan_rows[-1]), plan_rows[-1]

            # The durable escalation event landed with the surface and the cap
            # (the emitter writes the SERVER-level events.jsonl).
            events = [row for row in oracle.events(REASON_REVIEW_CYCLES_EXHAUSTED)
                      if str(row.get("surface") or "") == "plan_review"]
            assert events, oracle.events(REASON_REVIEW_CYCLES_EXHAUSTED)
            assert int(events[-1].get("cycles_paid") or 0) == 2, events[-1]
            review_script.assert_consumed()
        finally:
            server.stop()


# ===========================================================================
# S15 — commit triad+scope, ADVISORY enforcement class
# ===========================================================================

S12_DOC = "docs/notes/system_e2e_w3a_advisory.md"
S12_MSG = "docs: system_e2e w3a advisory-class smoke (doc-only)"
S12_SCRIPT = [
    {"tool": "write_file", "arguments": {"root": "system_repo", "path": S12_DOC,
        "content": "# w3a advisory-class smoke\n\nDoc-only change for the enforcement-class pin.\n",
    }},
    {"tool": "commit_reviewed", "arguments": {
        "commit_message": S12_MSG, "paths": [S12_DOC],
        "skip_advisory_review": True, "skip_tests": True,
        "goal": "Land the advisory-class smoke note despite a scripted red triad verdict.",
        "scope": f"{S12_DOC} only.",
    }},
]


@pytest.mark.integration
@pytest.mark.serial
def test_s15_advisory_class_red_verdict_recorded_and_commit_lands(e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s12")
    # One wave, one brief: the packet rows answer contract A (the scripted red
    # verdicts), the native row answers contract B clean; one critical anywhere
    # makes the wave red.
    review_script = ReviewScript({"triad_review": [W3A_TRIAD_RED] * len(KEYLESS_PACKET_ROWS)})
    feedback = {}
    decision = {"disposition": "rejected", "rationale": "I inspected the missing-marker criticism; this doc-only enforcement fixture intentionally retains the note and records my decision."}

    def continue_after_feedback(body):
        try:
            calls = {c["id"]: c["function"]["name"] for m in body["messages"] for c in m.get("tool_calls", [])}
            message = next(m for m in reversed(body["messages"]) if m.get("role") == "tool" and calls.get(m.get("tool_call_id")) == "commit_reviewed")
            feedback.update(json.loads(message["content"].split("\n", 1)[1]), head=_head(e2e_clone))
            critic = feedback["review_outcome"]
            assert feedback["head"] == head_before, "commit preceded feedback exposure"
            if critic["status"] == "reviewing":
                return _Again(S12_SCRIPT[1])  # collect the same custody, never a new paid panel
            assert critic["status"] == "reviewed" and critic["phase"] == "review_only" and critic["paid"]
            assert any(f.get("severity") == "critical" and "scripted critical finding (system_e2e w3a)" in f.get("reason", "") for f in critic["critical_findings"])
        except (AssertionError, KeyError, ValueError, StopIteration) as exc:
            return {"final": f"E2E_SCRIPT_ERROR: expected exposed critical commit feedback: {exc}"}
        return {"tool": "commit_reviewed", "arguments": {**S12_SCRIPT[1]["arguments"],
            "review_reference": feedback["review_reference"], "author_disposition": decision}}

    stub = _HoldingStubModel([*S12_SCRIPT, continue_after_feedback], review_script=review_script)
    with stub:
        settings = keyless_settings(stub, OUROBOROS_RUNTIME_MODE="advanced", OUROBOROS_REVIEW_ENFORCEMENT="advisory")
        server = start_server(e2e_clone, root, settings)
        try:
            head_before = _head(e2e_clone)
            task_id = submit_running(server, "Write the advisory-class note, inspect the critical feedback, explicitly choose whether to commit, then finish.")
            result = server.wait_task(task_id, timeout=600)
            assert result.get("status") == "completed", result
            assert "E2E_SCRIPT_ERROR" not in str(result), result
            oracle = ArtifactOracle(server.data_root)
            wait_durable_result(oracle, task_id)
            assert feedback["head"] == head_before and stub.script_consumed()
            assert _git_log_subjects(e2e_clone).count(S12_MSG) == 1
            task_drive = oracle.task_drive(task_id)
            # Loud Advisory remains durable: the original critical cause and counter.
            overrides = task_drive.events("review_advisory_override")
            assert overrides, "no review_advisory_override event in the task drive"
            assert overrides[-1].get("block_reason") == "critical_findings", overrides[-1]
            counter = json.loads((task_drive.data_root / "state" / "advisory_overrides.json").read_text(encoding="utf-8"))
            assert int(counter.get("count") or 0) >= 1, counter
            # The paid critic remains intact beside a separate unpaid author commit.
            attempts = [a for a in task_drive.advisory_review().get("attempts", []) if a.get("task_id") == task_id]
            assert "scripted critical finding (system_e2e w3a)" in json.dumps(attempts), attempts
            reference = feedback["review_reference"]
            critic = next(a for a in attempts if a["attempt"] == reference["attempt"])
            succeeded = next(a for a in attempts if a["status"] == "succeeded")
            assert critic["status"] == "reviewed" and critic["triad_raw_results"] == feedback["review_outcome"]["triad_raw_results"]
            assert sum(bool(a.get("paid")) for a in attempts) == 1 and not succeeded["paid"]
            author = succeeded["author_disposition"]
            assert author["review_reference"] == reference
            assert author["subject_hash"] == succeeded["pre_review_fingerprint"] == succeeded["post_review_fingerprint"] == reference["pre_review_fingerprint"]
            assert author["enforcement"] == "advisory" and author["source"] == "author"
            assert {key: author[key] for key in decision} == decision and author["recorded_at"]
            assert len(_tool_rows(task_drive, "commit_reviewed")) >= 2
            # ONE paid wave over the whole pool: every packet row and the native row
            # once; the author's commit paid no second panel.
            kinds = stub.kinds()
            assert kinds.count("triad_review") == len(KEYLESS_PACKET_ROWS), kinds
            assert kinds.count("two_part_review") == len(KEYLESS_REVIEW_ROWS) - len(KEYLESS_PACKET_ROWS), kinds
            review_script.assert_consumed()
        finally:
            server.stop()


# ===========================================================================
# S16 — commit triad+scope, BLOCKING enforcement class + freshness staleness
# ===========================================================================

S13_DOC = "docs/notes/system_e2e_w3a_blocking.md"
S13_MSG = "docs: system_e2e w3a blocking-class smoke (doc-only)"


def _s13_commit_step() -> dict:
    return {"tool": "commit_reviewed", "arguments": {
        "commit_message": S13_MSG,
        "paths": [S13_DOC],
        "skip_advisory_review": True,
        "skip_tests": True,
        "goal": "Land the blocking-class smoke note through the full triad+scope organ.",
        "scope": f"{S13_DOC} only.",
    }}


S13_SCRIPT = [
    {"tool": "write_file", "arguments": {
        "root": "system_repo", "path": S13_DOC,
        "content": "# w3a blocking-class smoke\n\nFirst candidate — reviewers will block this.\n",
    }},
    _s13_commit_step(),   # red triad -> REVIEW_BLOCKED
    _s13_commit_step(),   # byte-identical resubmit -> IDENTICAL_DIFF_REFUSED (free)
    {"tool": "write_file", "arguments": {
        "root": "system_repo", "path": S13_DOC,
        "content": ("# w3a blocking-class smoke\n\nSecond candidate with the marker.\n"
                    "W3A_MARKER: verification marker line added per review.\n"),
    }},
    _s13_commit_step(),   # clean triad+scope -> commit lands
]


@pytest.mark.integration
@pytest.mark.serial
def test_s16_blocking_class_red_blocks_identical_refused_free_then_green_lands(
        e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s13")
    # Wave 1: the packet rows answer red (contract A) beside a clean native seat.
    review_script = ReviewScript({"triad_review": [W3A_TRIAD_RED] * len(KEYLESS_PACKET_ROWS)})
    stub = ScriptedStubModel(S13_SCRIPT, review_script=review_script)
    with stub:
        settings = keyless_settings(
            stub,
            OUROBOROS_RUNTIME_MODE="advanced",
            OUROBOROS_REVIEW_ENFORCEMENT="blocking",
        )
        server = start_server(e2e_clone, root, settings)
        try:
            head_before = _head(e2e_clone)
            task_id = submit_running(
                server, "Write the blocking-class note and land it via commit_reviewed, then finish.")
            result = server.wait_task(task_id, timeout=600)
            assert result.get("status") == "completed", result
            oracle = ArtifactOracle(server.data_root)
            wait_durable_result(oracle, task_id)
            task_drive = oracle.task_drive(task_id)

            # Tool-result truth in order: blocked -> identical-refused -> landed.
            commit_rows = _tool_rows(task_drive, "commit_reviewed")
            assert len(commit_rows) == 3, commit_rows
            first, second, third = (json.dumps(row) for row in commit_rows)
            assert "REVIEW_BLOCKED" in first, commit_rows[0]
            assert "scripted critical finding (system_e2e w3a)" in first, commit_rows[0]
            assert "IDENTICAL_DIFF_REFUSED" in second, commit_rows[1]
            assert "REVIEW_BLOCKED" not in second, commit_rows[1]
            assert "IDENTICAL_DIFF_REFUSED" not in third and "REVIEW_BLOCKED" not in third, commit_rows[2]

            # The commit landed exactly once, with the FIXED content, and the
            # blocked attempts moved HEAD not at all (one new commit total).
            log_subjects = _git_log_subjects(e2e_clone)
            assert log_subjects.count(S13_MSG) == 1, log_subjects
            committed = subprocess.run(
                ["git", "show", f"HEAD:{S13_DOC}"], cwd=str(e2e_clone),
                check=True, capture_output=True, text=True).stdout
            assert "W3A_MARKER" in committed, committed
            head_after = _head(e2e_clone)
            parent = subprocess.run(
                ["git", "rev-parse", "HEAD~1"], cwd=str(e2e_clone),
                check=True, capture_output=True, text=True).stdout.strip()
            assert head_after != head_before and parent == head_before, (
                head_before, head_after, parent)

            # Durable ledger: a VERDICT-blocked attempt (critical_findings) is
            # recorded; the identical resubmit paid nothing (two paid waves over
            # the whole pool: red wave + clean wave, none for the resubmit).
            attempts = task_drive.advisory_review().get("attempts") or []
            blocked = [a for a in attempts if isinstance(a, dict)
                       and a.get("block_reason") == "critical_findings"]
            assert blocked, attempts
            kinds = stub.kinds()
            assert kinds.count("triad_review") == 2 * len(KEYLESS_PACKET_ROWS), kinds
            assert kinds.count("two_part_review") == 2 * (len(KEYLESS_REVIEW_ROWS) - len(KEYLESS_PACKET_ROWS)), kinds
            review_script.assert_consumed()
        finally:
            server.stop()


# --- S16 post-verdict revalidation (private clone: the scenario mutates the
# staged index mid-review, which must never leak into the shared session clone).

S13B_DOC = "docs/notes/system_e2e_w3a_freshness.md"
S13B_MSG = "docs: system_e2e w3a freshness smoke (doc-only)"
S13B_JUNK = "w3a_freshness_junk.txt"


S13B_SCRIPT = [
    {"tool": "write_file", "arguments": {
        "root": "system_repo", "path": S13B_DOC,
        "content": "# w3a freshness smoke\n\nCandidate for the post-verdict revalidation.\n",
    }},
    # No preflight_reviewer: the commit records `preflight: not_performed` and goes
    # straight to the paid triad+scope wave the scope hook mutates under.
    {"tool": "commit_reviewed", "arguments": {
        "commit_message": S13B_MSG,
        "paths": [S13B_DOC],
        "skip_tests": True,
        "goal": "Land the freshness smoke note.",
        "scope": f"{S13B_DOC} only.",
    }},
]


@pytest.mark.integration
@pytest.mark.serial
def test_s16_rejects_post_verdict_mutation(tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s13b")
    clone = clone_repo(root)

    def _mutate_staged_tree_then_pass(_body):
        # The post-verdict freshness probe: stage NEW bytes while the paid
        # review wave is in flight (after the pre-dispatch fingerprint, before
        # settlement). The two-part verdict returned here is ALL-CLEAN — the
        # refusal below can only come from the freshness gate, never from the
        # verdicts.
        (clone / S13B_JUNK).write_text("staged mid-review to prove post-verdict freshness\n",
                                       encoding="utf-8")
        subprocess.run(["git", "add", S13B_JUNK], cwd=str(clone),
                       check=True, capture_output=True)
        return two_part_clean_text()

    review_script = ReviewScript({"two_part_review": [_mutate_staged_tree_then_pass]})
    stub = ScriptedStubModel(S13B_SCRIPT, review_script=review_script)
    with stub:
        settings = keyless_settings(
            stub,
            OUROBOROS_RUNTIME_MODE="advanced",
            OUROBOROS_REVIEW_ENFORCEMENT="blocking",
        )
        server = start_server(clone, root, settings)
        try:
            head_before = _head(clone)
            task_id = submit_running(
                server, "Run the freshness smoke: write the note, then try to commit; finish.")
            result = server.wait_task(task_id, timeout=600)
            assert result.get("status") == "completed", result
            oracle = ArtifactOracle(server.data_root)
            wait_durable_result(oracle, task_id)
            task_drive = oracle.task_drive(task_id)

            commit_rows = _tool_rows(task_drive, "commit_reviewed")
            assert len(commit_rows) == 1, commit_rows

            # All-clean verdicts for OTHER bytes are rejected — the typed
            # revalidation refusal, mismatch fingerprint status.
            reval_refusal = json.dumps(commit_rows[0])
            assert "REVIEW_REVALIDATION_FAILED" in reval_refusal, commit_rows[0]
            attempts = task_drive.advisory_review().get("attempts") or []
            reval = [a for a in attempts if isinstance(a, dict)
                     and a.get("block_reason") == "revalidation_failed"]
            assert reval, attempts
            assert reval[-1].get("fingerprint_status") == "mismatch", reval[-1]
            assert task_drive.events("reviewed_attempt_revalidation_failed"), (
                "typed revalidation event missing")

            # Nothing ever landed: HEAD did not move, the message is nowhere.
            assert _head(clone) == head_before
            assert S13B_MSG not in _git_log_subjects(clone)

            # Call accounting: no preflight was named, so no preflight seat ran;
            # exactly one wave over the whole pool — the packet rows and the
            # hooked native seat.
            kinds = stub.kinds()
            assert "advisory_review" not in kinds, kinds
            assert kinds.count("triad_review") == len(KEYLESS_PACKET_ROWS), kinds
            assert kinds.count("two_part_review") == len(KEYLESS_REVIEW_ROWS) - len(KEYLESS_PACKET_ROWS), kinds
            review_script.assert_consumed()
        finally:
            server.stop()


# ===========================================================================
# S17 — acceptance loop (required + blocking)
# ===========================================================================

S14_ANSWER_V1 = "Final answer: the summary is drafted (first pass)."
S14_ANSWER_V2 = "Final answer: the summary is complete. W3A_DONE"


_OWNER_SOURCE_RE = re.compile(r'"owner_source_sha256": "([0-9a-f]{64})"')


def _completion_step(full_answer: str, *, answer_form: str):
    """Select complete bytes or an offered answer hash after host feedback."""
    def step(body: dict) -> dict:
        text = body_text(body)
        assert any(row.get("function", {}).get("name") == "finish_task"
                   for row in body.get("tools", [])), "finish_task was not offered"
        arguments = {"action": "finish"}
        if answer_form == "answer":
            arguments["answer"] = full_answer
        else:
            selector = hashlib.sha256(full_answer.encode("utf-8")).hexdigest()
            assert selector in re.findall(r"(?:retained as |answer_sha256=)([0-9a-f]{64})", text), (
                "The complete answer must be offered before selecting its hash", selector)
            arguments["answer_sha256"] = selector
        found = _OWNER_SOURCE_RE.findall(text)
        if found:
            arguments["acceptance_subject"] = {"owner_source_sha256": found[-1]}
        return {"tool": "finish_task", "arguments": arguments}
    return step


def _s14_settings(stub) -> dict:
    return keyless_settings(
        stub,
        OUROBOROS_TASK_REVIEW_MODE="required",
        OUROBOROS_REVIEW_ENFORCEMENT="blocking",
    )


def _keep_until_acceptance_settled(full_answer: str, *, wave_ordinal: int, answer_form: str):
    """Keep the same answer through quorum wakes until the intended wave settles.

    A final-slot notification can arrive during Main's response to the quorum
    wake. The host then drains it in another round, reusing the paid verdict.
    That round still belongs to this phase, not the stub's exhausted fallback.
    """
    keep = _completion_step(full_answer, answer_form=answer_form)
    waits = {"rounds": 0}

    def step(body: dict):
        frames = re.findall(
            r"Acceptance review (task_acceptance:[0-9a-f]+): (\d+) of (\d+) reviewer slot\(s\)",
            body_text(body),
        )
        settled = {wave for wave, answered, total in frames if answered == total}
        if len(settled) >= wave_ordinal:
            return keep(body)
        waits["rounds"] += 1
        if waits["rounds"] > _WAIT_ROUNDS_MAX:
            return {"final": (
                "E2E_SCRIPT_ERROR: acceptance wave "
                f"{wave_ordinal} did not settle after {waits['rounds']} round(s)")}
        return _Again(keep(body))

    return step


@pytest.mark.integration
@pytest.mark.serial
@pytest.mark.parametrize("answer_form", ["answer", "answer_sha256"])
def test_s17_acceptance_reject_rework_accept(e2e_clone, tmp_path_factory, answer_form):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s14")
    review_script = ReviewScript({
        "acceptance": [W3A_ACCEPT_REJECT] * 3 + [W3A_ACCEPT_PASS] * 3,
    })
    # Both answer forms retain V2 through separate quorum/final-slot wakes; repeated collection must
    # not buy a third panel or exhaust the script. A new hash is selectable only once the whole response was held.
    stub = _HoldingStubModel(
        [{"final": S14_ANSWER_V1}, *([{"final": S14_ANSWER_V2}] if answer_form == "answer_sha256" else []),
         _completion_step(S14_ANSWER_V2, answer_form=answer_form),
         _keep_until_acceptance_settled(S14_ANSWER_V2, wave_ordinal=2, answer_form=answer_form)],
        review_script=review_script,
    )
    with stub:
        server = start_server(e2e_clone, root, _s14_settings(stub))
        try:
            task_id = submit_running(
                server, "Summarize the w3a acceptance smoke and finish with the W3A_DONE marker.")
            result = server.wait_task(task_id, timeout=600)
            assert result.get("status") == "completed", result
            oracle = ArtifactOracle(server.data_root)
            stored = wait_durable_result(oracle, task_id)

            # The dialogue converged on the REWORKED answer, accepted clean.
            assert "W3A_DONE" in str(stored.get("result") or ""), stored.get("result")
            review_axis = (stored.get("outcome_axes") or {}).get("review") or {}
            decision = review_axis.get("acceptance_decision") or {}
            assert decision.get("status") == "accepted", review_axis
            signals = review_axis.get("aggregate_signals") or []
            assert "PASS" in signals and "FAIL" not in signals, review_axis

            # Paid-identity invariant: BOTH candidate identities were paid for —
            # the durable wallet carries two distinct claims.
            wallet = stored.get("task_acceptance_review_accounting") or {}
            claims = wallet.get("claims_by_binding") or {}
            assert len(claims) == 2, wallet

            # Exactly two panels of three slots each hit the model; the
            # scripted reject AND accept rounds were fully served.
            assert stub.kinds().count("acceptance") == 6, stub.kinds()
            review_script.assert_consumed()
        finally:
            server.stop()


@pytest.mark.integration
@pytest.mark.serial
@pytest.mark.parametrize("answer_form", ["answer", "answer_sha256"])
def test_s17_acceptance_identical_rework_is_free_replay_refusal(
        e2e_clone, tmp_path_factory, answer_form):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s14b")
    review_script = ReviewScript({"acceptance": [W3A_ACCEPT_REJECT] * 3})
    # Explicitly select the unchanged answer after rejection; both selection
    # forms must replay its paid verdict at $0 without a new panel.
    stub = _HoldingStubModel(
        [{"final": S14_ANSWER_V1},
         _keep_until_acceptance_settled(S14_ANSWER_V1, wave_ordinal=1, answer_form=answer_form)],
        review_script=review_script,
    )
    with stub:
        server = start_server(e2e_clone, root, _s14_settings(stub))
        try:
            task_id = submit_running(
                server, "Summarize the w3a acceptance smoke and finish with the W3A_DONE marker.")
            server.wait_task(task_id, timeout=600)  # terminal status asserted durably below
            oracle = ArtifactOracle(server.data_root)
            stored = wait_durable_result(oracle, task_id)

            # Free-replay invariant ($0-refusal class): the unchanged paid
            # identity is refused WITHOUT a second paid panel — exactly one
            # panel of three calls ever hit the model, and exactly one claim
            # sits on the durable wallet.
            assert stub.kinds().count("acceptance") == 3, stub.kinds()
            wallet = stored.get("task_acceptance_review_accounting") or {}
            claims = wallet.get("claims_by_binding") or {}
            assert len(claims) == 1, wallet

            # The terminal is the honest typed refusal, not a silent accept.
            review_axis = (stored.get("outcome_axes") or {}).get("review") or {}
            decision = review_axis.get("acceptance_decision") or {}
            assert decision.get("status") == "finalized_unaccepted", review_axis
            assert "identical" in str(decision.get("reason") or ""), decision
            review_script.assert_consumed()
        finally:
            server.stop()
