"""S14-S17 — Ф4 wave 3a of the deep-integration suite (v7next plan §8).

Review surfaces and acceptance, keyless throughout, on the wave-1/2 skeleton.
Every scenario asserts DURABLE artifacts (never an HTTP 200 alone, never a
harness exit code) and synchronizes by durable-event polling:

* S14 — PLAN REVIEW: a scripted task drives ``plan_task`` through a real
  REVISE→ACCEPT cycle (cycle 1: every triad slot returns a blocking finding →
  REVISE_PLAN; cycle 2: a CHANGED spec → all-clean → GREEN closed), the durable
  chronicle is honest (``plan_review_state`` on the stored task row: paid-cycle
  count, wave aggregates; immutable per-wave artifacts with the exact reviewer
  outputs), and the shared owner cycle cap (``OUROBOROS_REVIEW_MAX_CYCLES``) is
  respected: a third paid cycle is refused with the typed
  ``PLAN_REVIEW_CYCLES_EXHAUSTED`` result at $0 (no reviewer dispatched) plus
  the durable ``review_cycles_exhausted`` escalation event.
* S15 — COMMIT TRIAD+SCOPE, ADVISORY enforcement class: the same red triad
  verdict that BLOCKS a blocking install is recorded and waved through — the
  commit lands, and the wave-through leaves the constitutional loud trace
  (``review_advisory_override`` event + ``state/advisory_overrides.json``
  counter + the verdicts on the durable commit-attempt row). Effect asserted BY
  CLASS (AGENTS.md directive): S15+S16 are the same organ under the two
  enforcement values.
* S16 — COMMIT TRIAD+SCOPE, BLOCKING enforcement class: a critical triad FAIL
  blocks the commit (repo HEAD does not move), a byte-identical resubmission is
  refused FREE with the typed ``IDENTICAL_DIFF_REFUSED`` (no reviewer paid
  twice for the same bytes), and a fixed diff passes clean review and lands.
  Plus the freshness stale-rejection contracts — the ACTUAL mechanics of this
  tree, both pinned live:
    (a) advisory freshness: a fresh ``preflight_review`` verdict is invalidated
        by a later worktree edit (``invalidate_advisory_after_mutation``:
        snapshot-hash + stale-from-edit mark), and ``commit_reviewed`` without
        the audited skip refuses with ``ADVISORY_PRE_REVIEW_REQUIRED`` naming
        the edit — $0, nothing dispatched;
    (b) post-verdict revalidation: the staged material is mutated WHILE the
        paid triad+scope wave is in flight (after the pre-dispatch fingerprint,
        before settlement) — verdicts come back all-clean and the commit is
        STILL refused (``REVIEW_REVALIDATION_FAILED``, block_reason
        ``revalidation_failed``, fingerprint_status ``mismatch``): a verdict
        for other bytes is never carried forward.
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

import json
import subprocess

import pytest

from tests.system_e2e.harness import (
    LANE_MOCK,
    NATIVE_EPISODE_MARKER,
    PLAN_REVIEW_MARKER,
    REVIEW_KINDS,
    ArtifactOracle,
    ReviewScript,
    ScriptedStubModel,
    classify_call,
    clone_repo,
    keyless_reviewer_slots,
    keyless_settings,
    require_lane,
    scope_clean_text,
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


def test_w3a_keyless_reviewer_slots_advisory_row_parses_under_the_trees_parser():
    from ouroboros.reviewer_slot_config import parse_reviewer_slots

    config = parse_reviewer_slots(keyless_reviewer_slots(advisory=True))
    assert config.advisory.enabled is True
    assert config.advisory.kind == "api_chat"
    assert config.advisory.target_id == "openai-compatible::mock-model"
    # The default form stays byte-compatible: no advisory key, shipped default row.
    config_default = parse_reviewer_slots(keyless_reviewer_slots())
    assert config_default.advisory.target_id == ""


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


def _tool_rows(oracle: ArtifactOracle, tool_name: str) -> list:
    return [row for row in oracle.tools_rows()
            if str(row.get("tool") or row.get("name") or "") == tool_name]


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


@pytest.mark.integration
@pytest.mark.serial
def test_s14_plan_review_revise_then_accept_cycle_with_honest_chronicle(
        e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s11")
    review_script = ReviewScript({"plan_review": [W3A_PLAN_RED] * 3})
    stub = ScriptedStubModel(
        [_plan_step(S11_SPEC_V1, "cycle 1"), _plan_step(S11_SPEC_V2, "cycle 2 — revised")],
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
            # bytes: one REVISE_PLAN wave, one GREEN wave. The task artifact
            # store lives under the SERVER data root (task_results/artifacts/),
            # not the task's forked drive.
            artifacts = sorted(
                (oracle.data_root / "task_results" / "artifacts").rglob("plan-review-wave-*.json"))
            assert len(artifacts) == 2, artifacts
            aggregates = []
            for path in artifacts:
                payload = json.loads(path.read_text(encoding="utf-8"))
                aggregates.append(str(payload.get("aggregate") or ""))
            assert aggregates == ["REVISE_PLAN", "GREEN"], aggregates

            # The REVISE wave chronicled the scripted finding honestly.
            revise_payload = json.loads(artifacts[0].read_text(encoding="utf-8"))
            revise_blob = json.dumps(revise_payload)
            assert "checkable marker" in revise_blob, revise_blob[:2000]

            # Exactly two paid waves of three slots each hit the model; the
            # scripted red round was fully served.
            assert stub.kinds().count("plan_review") == 6, stub.kinds()
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
    stub = ScriptedStubModel(
        [_plan_step(S11_SPEC_V1, "cycle 1"),
         _plan_step(S11_SPEC_V2, "cycle 2 — still open"),
         _plan_step(S11_SPEC_V3, "cycle 3 — must be refused by the cap")],
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
    {"tool": "write_file", "arguments": {
        "root": "system_repo", "path": S12_DOC,
        "content": "# w3a advisory-class smoke\n\nDoc-only change for the enforcement-class pin.\n",
    }},
    {"tool": "commit_reviewed", "arguments": {
        "commit_message": S12_MSG,
        "paths": [S12_DOC],
        "skip_advisory_review": True,
        "skip_tests": True,
        "goal": "Land the advisory-class smoke note despite a scripted red triad verdict.",
        "scope": f"{S12_DOC} only.",
    }},
]


@pytest.mark.integration
@pytest.mark.serial
def test_s15_advisory_class_red_verdict_recorded_and_commit_lands(
        e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s12")
    review_script = ReviewScript({"triad_review": [W3A_TRIAD_RED] * 3})
    stub = ScriptedStubModel(S12_SCRIPT, review_script=review_script)
    with stub:
        settings = keyless_settings(
            stub,
            OUROBOROS_RUNTIME_MODE="advanced",
            OUROBOROS_REVIEW_ENFORCEMENT="advisory",
        )
        server = start_server(e2e_clone, root, settings)
        try:
            task_id = submit_running(
                server, "Write the advisory-class note and land it via commit_reviewed, then finish.")
            result = server.wait_task(task_id, timeout=600)
            assert result.get("status") == "completed", result
            oracle = ArtifactOracle(server.data_root)
            wait_durable_result(oracle, task_id)

            # The commit LANDED despite the critical triad verdicts — the
            # advisory class waves through instead of blocking.
            assert S12_MSG in _git_log_subjects(e2e_clone)

            # The constitutional loud trace (BIBLE P3 "loud advisory"): the
            # typed override event AND the persistent counter file.
            task_drive = oracle.task_drive(task_id)
            overrides = task_drive.events("review_advisory_override")
            assert overrides, "no review_advisory_override event in the task drive"
            assert overrides[-1].get("block_reason") == "critical_findings", overrides[-1]
            counter_path = task_drive.data_root / "state" / "advisory_overrides.json"
            counter = json.loads(counter_path.read_text(encoding="utf-8"))
            assert int(counter.get("count") or 0) >= 1, counter

            # The verdicts themselves are durably recorded on the commit-attempt
            # ledger (state/advisory_review.json attempts).
            attempts = task_drive.advisory_review().get("attempts") or []
            attempt_blob = json.dumps(attempts)
            assert "scripted critical finding (system_e2e w3a)" in attempt_blob, (
                "red triad verdicts missing from the durable commit-attempt ledger")

            # Both organs actually ran on the stub.
            kinds = stub.kinds()
            assert kinds.count("triad_review") == 3, kinds
            assert kinds.count("scope_review") == 1, kinds
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
    review_script = ReviewScript({"triad_review": [W3A_TRIAD_RED] * 3})
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
            # recorded; the identical resubmit paid nothing (6 triad calls
            # total: red wave + clean wave, none for the resubmit).
            attempts = task_drive.advisory_review().get("attempts") or []
            blocked = [a for a in attempts if isinstance(a, dict)
                       and a.get("block_reason") == "critical_findings"]
            assert blocked, attempts
            kinds = stub.kinds()
            assert kinds.count("triad_review") == 6, kinds
            assert kinds.count("scope_review") == 2, kinds
            review_script.assert_consumed()
        finally:
            server.stop()


# --- S16 freshness stale-rejection (private clone: the scenario mutates the
# staged index mid-review, which must never leak into the shared session clone).

S13B_DOC = "docs/notes/system_e2e_w3a_freshness.md"
S13B_MSG = "docs: system_e2e w3a freshness smoke (doc-only)"
S13B_JUNK = "w3a_freshness_junk.txt"


def _s13b_commit_step(*, skip_advisory: bool) -> dict:
    return {"tool": "commit_reviewed", "arguments": {
        "commit_message": S13B_MSG,
        "paths": [S13B_DOC],
        "skip_advisory_review": skip_advisory,
        "skip_tests": True,
        "goal": "Land the freshness smoke note.",
        "scope": f"{S13B_DOC} only.",
    }}


S13B_SCRIPT = [
    {"tool": "write_file", "arguments": {
        "root": "system_repo", "path": S13B_DOC,
        "content": "# w3a freshness smoke\n\nCandidate reviewed by the advisory episode.\n",
    }},
    # The doc-only scope is named ALONE, with no VERSION: this step is also the
    # live proof of the doc-only carve in the advisory admission (owner 11A,
    # finding W3A-F1). Before the carve, `release_metadata_preflight` blocked
    # ANY changed set without VERSION in scope — including the doc-only diffs
    # the commit gate exempts — so this scenario had to name the UNCHANGED
    # VERSION to reach a real verdict at all, and every real install's doc-only
    # work degraded to the audited bypass.
    {"tool": "preflight_review", "arguments": {
        "commit_message": S13B_MSG, "skip_tests": True, "paths": [S13B_DOC],
    }},
    {"tool": "write_file", "arguments": {
        "root": "system_repo", "path": S13B_DOC,
        "content": "# w3a freshness smoke\n\nEDITED AFTER the advisory verdict — advisory is stale.\n",
    }},
    _s13b_commit_step(skip_advisory=False),  # -> ADVISORY_PRE_REVIEW_REQUIRED (stale from edit)
    _s13b_commit_step(skip_advisory=True),   # -> clean verdicts, then revalidation_failed
]


@pytest.mark.integration
@pytest.mark.serial
def test_s16_freshness_stale_rejection_advisory_edit_and_post_verdict_mutation(
        tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s13b")
    clone = clone_repo(root)

    def _mutate_staged_tree_then_pass(_body):
        # The post-verdict freshness probe: stage NEW bytes while the paid
        # review wave is in flight (after the pre-dispatch fingerprint, before
        # settlement). The scope verdict returned here is ALL-CLEAN — the
        # refusal below can only come from the freshness gate, never from the
        # verdicts.
        (clone / S13B_JUNK).write_text("staged mid-review to prove post-verdict freshness\n",
                                       encoding="utf-8")
        subprocess.run(["git", "add", S13B_JUNK], cwd=str(clone),
                       check=True, capture_output=True)
        return scope_clean_text()

    review_script = ReviewScript({"scope_review": [_mutate_staged_tree_then_pass]})
    stub = ScriptedStubModel(S13B_SCRIPT, review_script=review_script)
    with stub:
        settings = keyless_settings(
            stub,
            OUROBOROS_RUNTIME_MODE="advanced",
            OUROBOROS_REVIEW_ENFORCEMENT="blocking",
            OUROBOROS_REVIEWER_SLOTS=keyless_reviewer_slots(advisory=True),
        )
        server = start_server(clone, root, settings)
        try:
            head_before = _head(clone)
            task_id = submit_running(
                server, "Run the freshness smoke: preflight, edit, then try to commit; finish.")
            result = server.wait_task(task_id, timeout=600)
            assert result.get("status") == "completed", result
            oracle = ArtifactOracle(server.data_root)
            wait_durable_result(oracle, task_id)
            task_drive = oracle.task_drive(task_id)

            # The advisory episode ran keyless on the stub and came back fresh.
            preflight_rows = _tool_rows(task_drive, "preflight_review")
            assert len(preflight_rows) == 1, preflight_rows
            preflight_result = str(preflight_rows[0].get("result_preview") or "")
            assert '"status": "fresh"' in preflight_result, preflight_result

            # Contract (a): the edit AFTER the verdict invalidated the advisory
            # — the un-skipped commit is refused with the typed stale message
            # naming the worktree edit, and NO reviewer was paid for it.
            commit_rows = _tool_rows(task_drive, "commit_reviewed")
            assert len(commit_rows) == 2, commit_rows
            stale_refusal = json.dumps(commit_rows[0])
            assert "ADVISORY_PRE_REVIEW_REQUIRED" in stale_refusal, commit_rows[0]
            assert "worktree edit" in stale_refusal, commit_rows[0]

            # The durable advisory ledger shows the fresh run demoted to stale.
            # (The transient last_stale_from_edit mark is consumed by the later
            # audited-bypass run of step 5; the refusal text above and the
            # stale-status run row are the durable contract.)
            advisory_state = task_drive.advisory_review()
            runs = advisory_state.get("advisory_runs") or []
            assert runs, advisory_state
            statuses = {str(r.get("status") or "") for r in runs if isinstance(r, dict)}
            assert "stale" in statuses, runs

            # Contract (b): all-clean verdicts for OTHER bytes are rejected —
            # the typed revalidation refusal, mismatch fingerprint status.
            reval_refusal = json.dumps(commit_rows[1])
            assert "REVIEW_REVALIDATION_FAILED" in reval_refusal, commit_rows[1]
            attempts = advisory_state.get("attempts") or []
            reval = [a for a in attempts if isinstance(a, dict)
                     and a.get("block_reason") == "revalidation_failed"]
            assert reval, attempts
            assert reval[-1].get("fingerprint_status") == "mismatch", reval[-1]
            assert task_drive.events("reviewed_attempt_revalidation_failed"), (
                "typed revalidation event missing")

            # Nothing ever landed: HEAD did not move, the message is nowhere.
            assert _head(clone) == head_before
            assert S13B_MSG not in _git_log_subjects(clone)

            # Call accounting: one advisory episode; exactly one paid triad
            # wave + the hooked scope call (the stale refusal was $0).
            kinds = stub.kinds()
            assert kinds.count("advisory_review") == 1, kinds
            assert kinds.count("triad_review") == 3, kinds
            assert kinds.count("scope_review") == 1, kinds
            assert kinds.index("advisory_review") < kinds.index("triad_review"), kinds
            review_script.assert_consumed()
        finally:
            server.stop()


# ===========================================================================
# S17 — acceptance loop (required + blocking)
# ===========================================================================

S14_ANSWER_V1 = "Final answer: the summary is drafted (first pass)."
S14_ANSWER_V2 = "Final answer: the summary is complete. W3A_DONE"


def _s14_settings(stub) -> dict:
    return keyless_settings(
        stub,
        OUROBOROS_TASK_REVIEW_MODE="required",
        OUROBOROS_REVIEW_ENFORCEMENT="blocking",
    )


@pytest.mark.integration
@pytest.mark.serial
def test_s17_acceptance_reject_rework_accept(e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s14")
    review_script = ReviewScript({
        "acceptance": [W3A_ACCEPT_REJECT] * 3 + [W3A_ACCEPT_PASS] * 3,
    })
    stub = ScriptedStubModel(
        [{"final": S14_ANSWER_V1}, {"final": S14_ANSWER_V2}],
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
def test_s17_acceptance_identical_rework_is_free_replay_refusal(
        e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s14b")
    review_script = ReviewScript({"acceptance": [W3A_ACCEPT_REJECT] * 3})
    stub = ScriptedStubModel(
        [{"final": S14_ANSWER_V1}, {"final": S14_ANSWER_V1}],  # rework changes NOTHING
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
