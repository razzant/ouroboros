"""S15-S16 — Ф4 wave 3a of the deep-integration suite (v7next plan §8).

Review surfaces, keyless throughout, on the wave-1/2 skeleton.
Every scenario asserts DURABLE artifacts (never an HTTP 200 alone, never a
harness exit code) and synchronizes by durable-event polling:

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
* S17 (the acceptance loop) retired with the task-acceptance machine: a
  task's result is reviewed by ``review_change(subject="result")``.

Covered by the other waves (manifest in ``tests/system_e2e/harness.py``):
delegated transport S11-S12, skills lifecycle S13 (wave 3b), self-evolution
absorb and the update variations in wave 4. Still deferred: gateway/UI truth
(Playwright).
"""

from __future__ import annotations

import hashlib
import json
import pathlib
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
    plan_record.parse_reply for a plan ask (the reply contract: prose, no objection,
    nothing withdrawn), the advisory clean predicate (shared
    empty_array_is_verified_clean) for the native episode."""
    from ouroboros.plan_record import parse_reply
    from ouroboros.triad_review import empty_array_is_verified_clean

    _kind, plan_msg = scripted_completion(_plan_body(), 1, lambda _b: None, "x")
    assert _kind == "plan_review"
    reply, parse_error = parse_reply(plan_msg["content"])
    assert parse_error == "" and reply["objections"] == [] and reply["withdrawn"] == [] and reply["prose"]

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
    assert kind == "plan_review" and msg["content"].startswith('{"prose"')
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


_OWNER_SOURCE_RE = re.compile(r'"owner_source_sha256": "([0-9a-f]{64})"')


def _completion_step(full_answer: str, *, answer_form: str):
    """Select complete bytes or an offered answer hash after host feedback (shared with wave 5 and
    the waiting-controls browser test; the acceptance loop that first used it retired)."""
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
# Script holds: wait on the task's own id until a mailbox frame lands
# ===========================================================================


# The task identity the host states in every prompt (the scripted agent waits on
# its OWN id).
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

    A review released at the dispatch barrier needs a step that repeats until a
    precondition is visible in the transcript: the agent waits for the settlement
    frame in its mailbox and only then acts again. A blind
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
            feedback.update(json.loads(message["content"].split("\n", 1)[1]), head=_head(oracle.body_candidate()["path"]))
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
            oracle = ArtifactOracle(server.data_root)
            task_id = submit_running(server, "Write the advisory-class note, inspect the critical feedback, explicitly choose whether to commit, then finish.")
            result = server.wait_task(task_id, timeout=600)
            assert result.get("status") == "completed", result
            assert "E2E_SCRIPT_ERROR" not in str(result), result
            wait_durable_result(oracle, task_id)
            assert feedback["head"] == head_before and stub.script_consumed()
            candidate = oracle.body_candidate(task_id)
            assert _git_log_subjects(candidate["path"]).count(S12_MSG) == 1
            assert _head(candidate["path"]) in candidate["reviewed_commits"]
            assert _head(e2e_clone) == head_before and not (e2e_clone / S12_DOC).exists()
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
            candidate = oracle.body_candidate(task_id)
            candidate_path = pathlib.Path(candidate["path"])
            log_subjects = _git_log_subjects(candidate_path)
            assert log_subjects.count(S13_MSG) == 1, log_subjects
            committed = subprocess.run(
                ["git", "show", f"HEAD:{S13_DOC}"], cwd=str(candidate_path),
                check=True, capture_output=True, text=True).stdout
            assert "W3A_MARKER" in committed, committed
            head_after = _head(candidate_path)
            parent = subprocess.run(
                ["git", "rev-parse", "HEAD~1"], cwd=str(candidate_path),
                check=True, capture_output=True, text=True).stdout.strip()
            assert head_after != head_before and parent == head_before, (
                head_before, head_after, parent)
            assert head_after in candidate["reviewed_commits"] and candidate["base_sha"] == head_before
            assert _head(e2e_clone) == head_before and not (e2e_clone / S13_DOC).exists()

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
        candidate_path = pathlib.Path(oracle.body_candidate()["path"])
        (candidate_path / S13B_JUNK).write_text("staged mid-review to prove post-verdict freshness\n",
                                       encoding="utf-8")
        subprocess.run(["git", "add", S13B_JUNK], cwd=str(candidate_path),
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
            oracle = ArtifactOracle(server.data_root)
            task_id = submit_running(
                server, "Run the freshness smoke: write the note, then try to commit; finish.")
            result = server.wait_task(task_id, timeout=600)
            assert result.get("status") == "completed", result
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
            candidate = oracle.body_candidate(task_id)
            candidate_path = pathlib.Path(candidate["path"])
            assert _head(candidate_path) == candidate["base_sha"] == head_before
            assert S13B_MSG not in _git_log_subjects(candidate_path)
            assert (candidate_path / S13B_JUNK).is_file()
            assert _head(clone) == head_before and not (clone / S13B_JUNK).exists()

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
