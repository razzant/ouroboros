"""#1223/#1224 — the LOCAL, pre-binding acceptance failure and what it may claim.

Assembling the acceptance packet can fail before any reviewer exists. That
failure used to be identified by the exception's own text over an EMPTY binding
hash, so alternating error types read as new incidents and the host/author pair
re-offered the same failure forever; the author could only answer by going back
through the same broken builder; and the owner-facing final then borrowed a
reviewer's words for a review that never ran.

These pin: one identity per MATERIAL (never the owner transcript, so a status
question is the same incident; the working-tree bytes ARE material, so a repair
is observable), one exposure per REAL attempt bound to that attempt, an
informed Advisory finish / Blocking stop that needs neither a working builder
nor a computable fingerprint, one-use source-bound retries whose spent keys
stay spent, no synthetic reviewer record, prior paid runs untouched, a repeat
guard that precedes every fallible pre-binding step of the host pass, and a
final cause that states the local failure AND the rail for THIS attempt only.

Any credential-shaped literal here is synthetic.
"""

from __future__ import annotations

import json
import copy
import subprocess as sp
from types import SimpleNamespace as NS

import pytest

from tests._acceptance_preparation_helpers import _ctx, _expose, _fail, _repo, _raise_fingerprint

pytestmark = pytest.mark.serial  # Source identity exercises real Git processes.


# ── identity: the material, never the transcript, the prose, the count or the error ──


def test_identity_is_the_material_and_ignores_candidate_prose_and_tool_count(tmp_path):
    from ouroboros.acceptance_preparation import preparation_source_identity

    ctx = _ctx(tmp_path)
    trace = {"tool_calls": []}
    first = preparation_source_identity(ctx.tools._ctx, trace)
    assert first["known"] is True and first["identity"] != "unknown"
    # A different candidate answer and more (non-material) tool calls are not new material.
    trace["tool_calls"] = [{"tool": "read_file", "status": "ok"}]
    assert preparation_source_identity(ctx.tools._ctx, trace)["identity"] == first["identity"]
    # Main's explicit current criteria ARE new material.
    ctx.tools._ctx._delivery_effective_criteria = "the owner now also wants a table"
    assert preparation_source_identity(ctx.tools._ctx, trace)["identity"] != first["identity"]


def test_a_status_question_changes_the_owner_acknowledgement_but_never_the_incident(tmp_path):
    """The owner transcript is NOT identity. 'how is it going?' is a new owner
    message through the real recorder — the same incident, a new acknowledgement."""
    from ouroboros.acceptance_preparation import begin_preparation
    from ouroboros.loop_messages import _record_owner_directive, owner_source_sha256

    ctx = _ctx(tmp_path)
    tool_ctx = ctx.tools._ctx
    record = _fail(begin_preparation(ctx.llm_trace, tool_ctx))
    first_id, first_ack = record["incident_id"], record["owner_source_sha256"]
    assert first_ack == owner_source_sha256(tool_ctx)

    _record_owner_directive(tool_ctx, source="owner_followup", content="how is it going?")
    assert owner_source_sha256(tool_ctx) != first_ack        # the owner corpus really changed
    again = begin_preparation(ctx.llm_trace, tool_ctx)
    assert again["incident_id"] == first_id                  # …and the incident did not
    assert again["attempts"] == 1
    assert again["owner_source_sha256"] == owner_source_sha256(tool_ctx)   # kept beside it


def test_source_bytes_are_material_so_a_repair_is_observable_and_an_unchanged_tree_is_stable(tmp_path):
    from ouroboros.acceptance_preparation import preparation_source_identity

    repo = _repo(tmp_path)
    ctx = _ctx(tmp_path, repo_dir=repo)
    trace = {"tool_calls": []}
    (repo / "src.py").write_bytes(b"x = 2\n# caf\xe9\n")              # the failing non-UTF-8 edit
    broken = preparation_source_identity(ctx.tools._ctx, trace)
    assert broken["known"] is True
    assert preparation_source_identity(ctx.tools._ctx, trace)["identity"] == broken["identity"]
    (repo / "src.py").write_text("x = 2\n# cafe\n", encoding="utf-8")   # the repair: same size, new bytes
    repaired = preparation_source_identity(ctx.tools._ctx, trace)
    assert repaired["identity"] != broken["identity"]
    (repo / "scratch_test.py").write_text("def test_x(): pass\n", encoding="utf-8")   # a new untracked file
    assert preparation_source_identity(ctx.tools._ctx, trace)["identity"] != repaired["identity"]


def test_an_unreadable_repository_is_one_stable_unknown_that_reopens_when_readable(tmp_path):
    from ouroboros.acceptance_preparation import begin_preparation

    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / ".git").write_text("gitdir: /nonexistent/gitdir\n", encoding="utf-8")   # a broken gitfile: Git-required, unreadable
    ctx = _ctx(tmp_path, repo_dir=corrupt)
    first = dict(begin_preparation(ctx.llm_trace, ctx.tools._ctx))
    assert first["source_known"] is False and first["unknown_parts"] == ["repository_source"]
    assert dict(begin_preparation(ctx.llm_trace, ctx.tools._ctx))["incident_id"] == first["incident_id"]
    _fail(ctx.llm_trace["acceptance_preparation"])

    (corrupt / ".git").unlink()
    sp.run(["git", "init"], cwd=corrupt, check=True, capture_output=True)   # it becomes readable
    known = begin_preparation(ctx.llm_trace, ctx.tools._ctx)
    assert known["source_known"] is True
    assert known["incident_id"] != first["incident_id"]
    assert known["attempts"] == 0                               # a readable source reopens honestly
    assert known["history"][-1]["source_identity"] == "unknown"  # the old one stays


def test_a_plain_workspace_is_known_material_that_moves_only_with_its_own_bytes(tmp_path):
    """A proven plain folder (no `.git`, no `HEAD`) is readable material, not an
    unreadable repository: its bounded content identity is known and stable, a
    same-size edit inside it is observable, and a parent checkout around it is
    never part of it."""
    from ouroboros.acceptance_preparation import preparation_source_identity

    parent = _repo(tmp_path)
    plain = parent / "workspace"
    plain.mkdir()
    (plain / "notes.txt").write_bytes(b"draft caf\xe9\n")
    ctx = _ctx(tmp_path, repo_dir=plain)
    trace = {"tool_calls": []}
    first = preparation_source_identity(ctx.tools._ctx, trace)
    assert first["known"] is True and first["unknown_parts"] == []
    assert preparation_source_identity(ctx.tools._ctx, trace)["identity"] == first["identity"]
    (parent / "src.py").write_text("x = 2\n", encoding="utf-8")             # the parent checkout moved
    assert preparation_source_identity(ctx.tools._ctx, trace)["identity"] == first["identity"]
    (plain / "notes.txt").write_bytes(b"draft cafe\n")                       # same size, new bytes
    assert preparation_source_identity(ctx.tools._ctx, trace)["identity"] != first["identity"]


def test_an_unreadable_context_is_one_stable_unknown(tmp_path):
    from ouroboros.acceptance_preparation import begin_preparation

    class _Unreadable:
        def __getattr__(self, name):
            raise OSError("source unavailable")

    trace: dict = {"tool_calls": []}
    first = dict(begin_preparation(trace, _Unreadable()))
    second = dict(begin_preparation(trace, _Unreadable()))
    assert first["source_identity"] == second["source_identity"] == "unknown"
    assert first["incident_id"] == second["incident_id"]
    assert second["attempts"] == 0  # begin does not count attempts; failures do


def test_alternating_error_types_over_the_same_material_stay_one_incident(tmp_path):
    from ouroboros.acceptance_preparation import begin_preparation, preparation_blocked

    ctx = _ctx(tmp_path)
    record = _expose(_fail(begin_preparation(ctx.llm_trace, ctx.tools._ctx), TypeError("first shape")))
    first_id, first_attempts = record["incident_id"], record["attempts"]
    assert preparation_blocked(record) is True
    # A NEW round with a different exception type is the same incident: the
    # builder is not run again, so nothing can count a second attempt.
    same = begin_preparation(ctx.llm_trace, ctx.tools._ctx)
    assert same["incident_id"] == first_id and same["attempts"] == first_attempts
    assert preparation_blocked(same) is True


# ── exposure: a queued message is not a delivered one, and it binds the attempt ──


def test_feedback_is_exposed_only_when_the_carrying_request_received_a_response(tmp_path):
    from ouroboros.acceptance_preparation import (
        begin_preparation, offer_preparation_feedback, preparation_blocked, preparation_exposed,
    )
    from ouroboros.acceptance_settlement import expose_acceptance_feedback

    ctx = _ctx(tmp_path)
    record = _fail(begin_preparation(ctx.llm_trace, ctx.tools._ctx))
    assert offer_preparation_feedback(ctx, record) is True
    assert preparation_exposed(record) is False       # queued only
    assert preparation_blocked(record) is False
    carried = ctx.messages[-1]["review_feedback"][0]
    assert carried["outcome_incident_id"] == record["incident_id"]
    assert carried["outcome_incident_attempt"] == 1

    expose_acceptance_feedback(ctx.llm_trace, ctx.messages, ctx.task_id)
    assert preparation_exposed(record) is True        # the request came back answered
    assert preparation_blocked(record) is True
    outcome = ctx.llm_trace["acceptance_review_outcome"]
    assert outcome["incident_id"] == record["incident_id"] and outcome["incident_attempt"] == 1


def test_feedback_about_an_earlier_attempt_exposes_nothing_about_a_later_one(tmp_path):
    from ouroboros.acceptance_preparation import (
        begin_preparation, offer_preparation_feedback, preparation_blocked, preparation_exposed,
    )
    from ouroboros.acceptance_settlement import expose_acceptance_feedback

    ctx = _ctx(tmp_path)
    record = _fail(begin_preparation(ctx.llm_trace, ctx.tools._ctx))
    assert offer_preparation_feedback(ctx, record) is True     # carries attempt 1
    _fail(record)                                              # a second real attempt failed
    assert record["attempts"] == 2
    expose_acceptance_feedback(ctx.llm_trace, ctx.messages, ctx.task_id)
    assert preparation_exposed(record) is False                # attempt 2 was never disclosed
    assert preparation_blocked(record) is False


def test_a_rephrased_final_or_status_question_buys_no_second_feedback(tmp_path):
    from ouroboros.acceptance_preparation import begin_preparation, offer_preparation_feedback
    from ouroboros.acceptance_settlement import expose_acceptance_feedback

    ctx = _ctx(tmp_path)
    record = _fail(begin_preparation(ctx.llm_trace, ctx.tools._ctx))
    assert offer_preparation_feedback(ctx, record) is True
    expose_acceptance_feedback(ctx.llm_trace, ctx.messages, ctx.task_id)
    before = len(ctx.messages)
    record["failure_kind"] = "ValueError"     # a different exception type changes nothing
    assert offer_preparation_feedback(ctx, record) is False
    assert len(ctx.messages) == before


def test_the_feedback_message_never_claims_a_reviewer_ran(tmp_path):
    from ouroboros.acceptance_preparation import begin_preparation, offer_preparation_feedback

    ctx = _ctx(tmp_path)
    record = _fail(begin_preparation(ctx.llm_trace, ctx.tools._ctx))
    offer_preparation_feedback(ctx, record)
    body = ctx.messages[-1]["content"]
    assert "dispatched no new reviewer" in body
    assert "earlier reviewer records, verdicts and costs are retained" in body
    assert record["incident_id"] in body and "host attempt 1" in body
    assert ctx.llm_trace["acceptance_decision"]["reason"] == "acceptance_preparation_failed"


# ── one-use, source-bound retry ────────────────────────────────────────────────


def test_an_explicit_source_bound_retry_grants_exactly_one_attempt(tmp_path):
    from ouroboros.acceptance_preparation import (
        begin_preparation, consume_retry, preparation_blocked, record_retry_intent,
    )

    ctx = _ctx(tmp_path)
    record = _expose(_fail(begin_preparation(ctx.llm_trace, ctx.tools._ctx)))
    assert preparation_blocked(record) is True
    from tests._acceptance_preparation_helpers import _receipt_retry
    intent = _receipt_retry(ctx, record, "the unreadable source file was restored")
    granted = record_retry_intent(ctx.llm_trace, intent, ctx.tools._ctx)
    assert granted["basis"] == "repair_evidence" and granted["consumed"] is False
    # Re-delivering the SAME declaration is idempotent: still one attempt.
    assert record_retry_intent(ctx.llm_trace, dict(intent), ctx.tools._ctx)["key"] == granted["key"]
    assert consume_retry(record) is True
    assert preparation_blocked(record) is False
    assert consume_retry(record) is False       # one use only


def test_a_spent_retry_declaration_stays_spent_a_b_a(tmp_path):
    """The record keeps EVERY key it granted: after A and B were consumed, A
    again buys nothing, so the same two sentences cannot be alternated forever."""
    from ouroboros.acceptance_preparation import (
        begin_preparation, consume_retry, preparation_blocked, record_retry_intent,
    )

    ctx = _ctx(tmp_path)
    record = _expose(_fail(begin_preparation(ctx.llm_trace, ctx.tools._ctx)))
    from tests._acceptance_preparation_helpers import _receipt_retry
    a = _receipt_retry(ctx, record, "restored the file")
    b = {"incident_id": record["incident_id"], "basis": "owner_retry", "rationale": "the owner asked",
         "owner_source_sha256": record["owner_source_sha256"]}
    assert record_retry_intent(ctx.llm_trace, a, ctx.tools._ctx) and consume_retry(record) is True
    _expose(_fail(record))
    assert record_retry_intent(ctx.llm_trace, b, ctx.tools._ctx) and consume_retry(record) is True
    _expose(_fail(record))
    assert record_retry_intent(ctx.llm_trace, dict(a), ctx.tools._ctx) == {}
    assert consume_retry(record) is False
    assert preparation_blocked(record) is True


def test_a_retry_naming_another_incident_or_no_basis_grants_nothing(tmp_path):
    from ouroboros.acceptance_preparation import begin_preparation, record_retry_intent

    ctx = _ctx(tmp_path)
    record = _fail(begin_preparation(ctx.llm_trace, ctx.tools._ctx))
    assert record_retry_intent(ctx.llm_trace, {
        "incident_id": "acceptance-preparation:someone-else",
        "basis": "owner_retry", "rationale": "please"}, ctx.tools._ctx) == {}
    assert record_retry_intent(ctx.llm_trace, {
        "incident_id": record["incident_id"], "basis": "because_i_said_so",
        "rationale": "please"}, ctx.tools._ctx) == {}
    assert "retry" not in record


def test_retry_requires_exposure_of_the_current_attempt(tmp_path):
    from ouroboros.acceptance_preparation import (
        begin_preparation, record_preparation_success, record_retry_intent,
    )

    ctx = _ctx(tmp_path)
    record = _fail(begin_preparation(ctx.llm_trace, ctx.tools._ctx))
    intent = {"incident_id": record["incident_id"], "basis": "owner_retry",
              "rationale": "retry the disclosed failure", "owner_source_sha256": record["owner_source_sha256"]}
    assert record_retry_intent(ctx.llm_trace, intent, ctx.tools._ctx) == {}
    # A response to attempt 1 does not expose the failure of attempt 2.
    _expose(record)
    _fail(record)
    assert record_retry_intent(ctx.llm_trace, intent, ctx.tools._ctx) == {}
    _expose(record)
    record_preparation_success(record)
    assert record_retry_intent(ctx.llm_trace, intent, ctx.tools._ctx) == {}  # No advance retry of a resolved incident.
    assert "retry" not in record


def test_an_ordinary_nomination_does_not_reset_the_guard(tmp_path):
    """A plain re-nomination is not a retry declaration: only the explicit,
    source-bound one is (owner decision 2, 22.09 23:33)."""
    from ouroboros.acceptance_preparation import begin_preparation, preparation_blocked
    from ouroboros.loop_tool_execution import process_tool_results

    ctx = _ctx(tmp_path)
    record = _expose(_fail(begin_preparation(ctx.llm_trace, ctx.tools._ctx)))
    process_tool_results(
        [{"fn_name": "task_acceptance_review", "tool_call_id": "c1", "is_error": False,
          "result": '{"status": "deferred_to_host_acceptance", "authoritative": false,'
                    ' "request": {"surface": "task_acceptance", "task_id": "root-delivery"}}',
          "args_for_log": {}, "tool_args": {}, "result_meta": {"status": "ok"}}],
        [], ctx.llm_trace, emit_progress=lambda _m, *, incident=None: None,
    )
    assert preparation_blocked(record) is True
    assert "retry" not in record


# ── the author decides without the broken builder OR a computable fingerprint ──


def test_an_informed_author_can_finish_or_stop_without_a_builder_or_a_fingerprint(tmp_path, monkeypatch):
    from ouroboros import loop as loop_mod
    from ouroboros.acceptance_preparation import begin_preparation, finish_exposed_preparation_author
    from ouroboros.loop_acceptance import merge_agent_acceptance_stance

    for action, expected in (("finish", "author_finish"), ("stop", "author_stop")):
        root = tmp_path / action
        root.mkdir(parents=True, exist_ok=True)
        ctx = _ctx(root)
        record = _expose(_fail(begin_preparation(ctx.llm_trace, ctx.tools._ctx)))
        _raise_fingerprint(monkeypatch)   # from here on nothing may need it
        monkeypatch.setattr(loop_mod, "get_review_enforcement",
                            lambda: "advisory" if action == "finish" else "blocking")
        merge_agent_acceptance_stance(ctx.llm_trace, {
            "disposition": "partial", "rationale": "delivering what is verified",
            "explicit_finish": True, "author_action": action,
        }, ctx.tools._ctx)
        intent = ctx.llm_trace["acceptance_decision"]["agent_finish_intent"]
        assert intent["preparation_identity"] == record["source_identity"]
        assert intent["incident_attempt"] == 1 and "evidence_fingerprint" not in intent

        assert finish_exposed_preparation_author(ctx, record) is True
        decision = ctx.llm_trace["acceptance_decision"]
        assert decision["reason"] == expected
        assert decision["status"] == "finalized_unaccepted"      # never a reviewer PASS
        assert decision["reviewer_signal"] == ""                 # no reviewer to signal
        assert decision["acceptance_incident"]["incident_id"] == record["incident_id"]
        assert decision["acceptance_incident"]["attempts"] == 1
        assert decision["author_disposition"]["subject_hash"] == f"{record['incident_id']}:attempt-1"


def test_a_blocking_install_may_only_stop_and_a_stance_for_other_material_or_attempt_is_refused(tmp_path, monkeypatch):
    from ouroboros import loop as loop_mod
    from ouroboros.acceptance_preparation import begin_preparation, finish_exposed_preparation_author

    ctx = _ctx(tmp_path)
    record = _expose(_fail(begin_preparation(ctx.llm_trace, ctx.tools._ctx)))
    monkeypatch.setattr(loop_mod, "get_review_enforcement", lambda: "blocking")
    informed = {"author_action": "finish", "preparation_identity": record["source_identity"],
                "incident_id": record["incident_id"], "incident_attempt": 1}
    ctx.llm_trace["acceptance_decision"] = {
        "agent_disposition": "accepted", "agent_rationale": "ship it", "agent_finish_intent": informed,
    }
    # Blocking grants no advancement without a verdict: only an explicit stop.
    assert finish_exposed_preparation_author(ctx, record) is False
    # A stance bound to OTHER material is not an informed decision about this one…
    ctx.llm_trace["acceptance_decision"]["agent_finish_intent"] = {**informed, "author_action": "stop",
                                                                   "preparation_identity": "some-other-material"}
    assert finish_exposed_preparation_author(ctx, record) is False
    # …and neither is a stance informed about an EARLIER attempt of this incident.
    _expose(_fail(record))
    ctx.llm_trace["acceptance_decision"]["agent_finish_intent"] = {**informed, "author_action": "stop"}
    assert finish_exposed_preparation_author(ctx, record) is False
    ctx.llm_trace["acceptance_decision"]["agent_finish_intent"] = {**informed, "author_action": "stop",
                                                                   "incident_attempt": 2}
    assert finish_exposed_preparation_author(ctx, record) is True


def _tool_ctx(tmp_path, **fields):
    values = dict(drive_root=str(tmp_path), drive_logs=lambda: tmp_path / "logs",
                  task_id="root", root_task_id="root", task_metadata={"root_task_id": "root"},
                  task_contract={})
    values.update(fields)
    return NS(**values)


def test_the_root_nomination_never_runs_the_builder_and_records_the_stance(tmp_path, monkeypatch):
    """tools/review.py used to build evidence BEFORE returning the nomination, so
    a broken builder took the author's ability to say "stop" with it. The root
    nomination now returns first: the builder is called ZERO times."""
    import ouroboros.review_evidence as re_mod
    from ouroboros.tools.review import _handle_task_acceptance_review

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "auto")
    calls = []

    def _explode(*_a, **_k):
        calls.append(1)
        raise RuntimeError("packet assembly failed")

    monkeypatch.setattr(re_mod, "build_task_acceptance_evidence", _explode)
    payload = json.loads(_handle_task_acceptance_review(
        _tool_ctx(tmp_path), claim="done", goal="g", agent_disposition="partial",
        rationale="stopping honestly", author_action="stop",
        evidence={"repo_diff": "the agent's own diff", "notes": "n"},
    ))
    assert calls == []                                              # the builder never ran
    assert payload["status"] == "deferred_to_host_acceptance" and payload["authoritative"] is False
    assert payload["agent_decision"]["author_action"] == "stop"
    assert payload["agent_decision"]["disposition"] == "partial"
    assert "acceptance_retry" not in payload  # Retry and a terminal stance are mutually exclusive.
    supplied = payload["agent_supplied"]                            # the builder's own normalization
    assert supplied["agent_supplied_repo_diff"] == "the agent's own diff" and "repo_diff" not in supplied
    assert supplied["acceptance_request"]["claim"] == "done"
    assert len(payload["evidence_revision"]) == 64


def test_action_only_nomination_does_not_invent_partial_stance(tmp_path, monkeypatch):
    from ouroboros.tools.review import _handle_task_acceptance_review

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "auto")
    payload = json.loads(_handle_task_acceptance_review(
        _tool_ctx(tmp_path), claim="saved result", goal="deliver result",
        rationale="Informed advisory finish with open critic notes", author_action="finish",
    ))
    assert payload["status"] == "deferred_to_host_acceptance"
    assert payload["agent_decision"]["disposition"] == ""
    assert payload["agent_decision"]["author_action"] == "finish"
    assert payload["agent_decision"]["explicit_finish"] is True


def test_the_child_path_still_builds_its_packet_and_a_broken_builder_still_raises(tmp_path, monkeypatch):
    import ouroboros.review_evidence as re_mod
    from ouroboros.tools.review import _handle_task_acceptance_review

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "auto")
    calls = []

    def _explode(*_a, **_k):
        calls.append(1)
        raise RuntimeError("packet assembly failed")

    monkeypatch.setattr(re_mod, "build_task_acceptance_evidence", _explode)
    child = _tool_ctx(tmp_path, task_id="child", parent_task_id="root", delegation_role="subagent")
    with pytest.raises(RuntimeError, match="packet assembly failed"):
        _handle_task_acceptance_review(child, claim="done", goal="g")
    assert calls == [1]


def test_a_malformed_retry_declaration_is_a_typed_argument_error(tmp_path, monkeypatch):
    from ouroboros.tools.review import _handle_task_acceptance_review

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "auto")
    out = _handle_task_acceptance_review(
        _tool_ctx(tmp_path), claim="done", goal="g",
        acceptance_retry={"incident_id": "", "basis": "owner_retry", "rationale": ""},
    )
    assert "TOOL_ARG_ERROR" in str(out)


# ── what the failure may NOT do to the record ─────────────────────────────────


def test_a_local_failure_records_no_reviewer_and_preserves_prior_paid_runs(tmp_path, monkeypatch):
    from ouroboros import loop as loop_mod
    from ouroboros.acceptance_preparation import begin_preparation, record_local_preparation_failure

    ctx = _ctx(tmp_path)
    paid = {
        "authority": "host_root", "aggregate_signal": "FAIL", "binding_hash": "b1",
        "paid_identity": "p1", "panel_id": "panel_b1", "cost_usd": 0.42,
        "actors": [{"slot_id": "s1", "operation_state": "in_flight"}],
        "request": {"surface": "task_acceptance"},
    }
    ctx.llm_trace["review_runs"] = [paid]
    before = copy.deepcopy(ctx.llm_trace["review_runs"])
    monkeypatch.setattr(loop_mod, "get_review_enforcement", lambda: "blocking")
    begin_preparation(ctx.llm_trace, ctx.tools._ctx)
    assert record_local_preparation_failure(ctx, RuntimeError("builder exploded")) is True  # one reaction
    from ouroboros.review_projection import publish_acceptance_checkpoint

    publish_acceptance_checkpoint(ctx.tools._ctx, ctx.llm_trace)
    assert ctx.llm_trace["review_runs"] == before
    assert paid["aggregate_signal"] == "FAIL" and paid["cost_usd"] == 0.42
    assert paid["actors"][0]["operation_state"] == "in_flight"
    assert ctx.llm_trace["acceptance_preparation"]["attempts"] == 1


def test_the_incident_identity_is_not_the_binding_or_the_paid_identity(tmp_path):
    from ouroboros.acceptance_preparation import begin_preparation

    ctx = _ctx(tmp_path)
    record = begin_preparation(ctx.llm_trace, ctx.tools._ctx)
    assert record["incident_id"].startswith("acceptance-preparation:")
    assert record["incident_id"] != ctx.review_binding["binding_hash"]
    assert record["source_identity"] != ctx.review_binding.get("paid_identity")


def test_a_successful_preparation_resolves_the_incident_and_keeps_its_history(tmp_path):
    from ouroboros.acceptance_preparation import (
        begin_preparation, close_local_preparation, consume_retry, incident_projection,
        record_retry_intent,
    )

    rows = []
    ctx = _ctx(tmp_path)
    ctx.emit_progress = lambda text, *, incident=None: rows.append(text)
    record = _expose(_fail(begin_preparation(ctx.llm_trace, ctx.tools._ctx)))
    # The retry reopened the record (status open): the resolution must still
    # see the failure it recovered from, not only a still-failed record.
    from tests._acceptance_preparation_helpers import _receipt_retry
    record_retry_intent(ctx.llm_trace, _receipt_retry(ctx, record, "restored"), ctx.tools._ctx)
    assert consume_retry(record) is True and record["status"] == "open"
    close_local_preparation(ctx, record)
    assert record["status"] == "resolved"
    assert record["history"][-1]["attempts"] == 1        # the failure is not erased
    assert rows == ["Acceptance evidence assembled after 1 failed host attempt; "
                    "the earlier failure stays in this task's history."]
    projection = incident_projection(record)
    assert projection["status"] == "resolved" and projection["prior_incidents"] == 1


def test_a_preparation_that_never_failed_is_no_incident_at_all(tmp_path):
    """The ordinary successful baseline has zero attempts: it projects nothing,
    publishes nothing and says nothing — there is no incident to show."""
    from ouroboros.acceptance_preparation import (
        begin_preparation, close_local_preparation, incident_projection,
    )
    from ouroboros.review_projection import publish_acceptance_checkpoint
    from ouroboros.task_results import load_task_result

    rows = []
    ctx = _ctx(tmp_path)
    ctx.emit_progress = lambda text, *, incident=None: rows.append(text)
    record = begin_preparation(ctx.llm_trace, ctx.tools._ctx)
    close_local_preparation(ctx, record)
    assert record["status"] == "resolved" and record["attempts"] == 0
    assert incident_projection(record) == {} and rows == []
    publish_acceptance_checkpoint(ctx.tools._ctx, ctx.llm_trace, task_id=ctx.task_id, drive_root=tmp_path)
    result = load_task_result(tmp_path, ctx.task_id) or {}
    assert (result.get("review_projection") or {}).get("acceptance_incident") is None


def test_each_real_attempt_states_one_plain_host_progress_line(tmp_path):
    """The owner-facing carrier is the review projection; the host's progress
    line is an ordinary note through the ONE progress ABI (text, incident=) —
    no placement kwargs, no retry on a sink that lacks them, no toast."""
    from ouroboros.acceptance_preparation import (
        begin_preparation, open_local_preparation, record_local_preparation_failure,
    )

    rows = []
    ctx = _ctx(tmp_path)
    ctx.emit_progress = lambda text, *, incident=None: rows.append((text, incident))
    begin_preparation(ctx.llm_trace, ctx.tools._ctx)
    record_local_preparation_failure(ctx, RuntimeError("builder exploded"))
    assert len(rows) == 1 and rows[0][1] is None
    assert "host attempt 1" in rows[0][0] and "No new reviewer was dispatched" in rows[0][0]
    open_local_preparation(ctx)            # a replayed round adds no second line
    assert len(rows) == 1


# ── the owner-facing truth ────────────────────────────────────────────────────


def test_the_final_cause_states_the_local_failure_and_the_rail_together(tmp_path):
    from ouroboros.project_dialogue import TASK_CAUSE_PHRASES, _completion_verdict

    incident = {"incident_id": "acceptance-preparation:abc", "status": "failed",
                "stage": "preparation", "attempts": 1}
    record = {
        "status": "completed", "reason_code": "budget_exhausted",
        "outcome_axes": {
            "execution": {"status": "degraded"},
            "review": {"status": "degraded", "acceptance_decision": {
                "status": "finalized_unaccepted", "reason": "acceptance_preparation_failed",
                "acceptance_incident": incident}},
        },
    }
    verdict = _completion_verdict(record, {})
    assert TASK_CAUSE_PHRASES["acceptance_preparation_failed"][:-1] in verdict
    assert TASK_CAUSE_PHRASES["budget_exhausted"] in verdict
    assert "this preparation attempt dispatched no new reviewers" in verdict
    assert "rework" not in verdict and "no reviewer ever saw" not in verdict


def test_a_genuine_reviewer_fail_is_not_replaced_by_the_incident(tmp_path):
    from ouroboros.project_dialogue import TASK_CAUSE_PHRASES, _completion_verdict

    record = {
        "status": "completed",
        "outcome_axes": {"review": {"acceptance_decision": {
            "status": "finalized_unaccepted", "reason": "reviewer_fail_no_capsule",
            "acceptance_incident": {"incident_id": "acceptance-preparation:abc",
                                    "status": "failed", "attempts": 1}}}},
    }
    verdict = _completion_verdict(record, {})
    assert TASK_CAUSE_PHRASES["reviewer_fail_no_capsule"][:-1] in verdict
    assert TASK_CAUSE_PHRASES["acceptance_preparation_failed"] in verdict


def test_a_resolved_incident_states_nothing_beside_the_answer(tmp_path):
    from ouroboros.project_dialogue import _completion_verdict

    record = {
        "status": "completed",
        "outcome_axes": {"review": {"acceptance_decision": {
            "status": "finalized_unaccepted", "reason": "author_finish",
            "acceptance_incident": {"incident_id": "acceptance-preparation:abc",
                                    "status": "resolved", "attempts": 1}}}},
    }
    assert "could not assemble" not in _completion_verdict(record, {})


def test_a_forced_rail_over_a_local_failure_keeps_its_own_cause(tmp_path):
    """The dangling revision came from the HOST's failure, so terminalizing it
    as "the requested rework never happened" would invent a reviewer."""
    from ouroboros.loop_acceptance import terminalize_dangling_revision

    trace = {"acceptance_decision": {
        "status": "revision_requested", "reason": "acceptance_preparation_failed",
        "acceptance_incident": {"incident_id": "acceptance-preparation:abc",
                                "status": "failed", "attempts": 1}}}
    assert terminalize_dangling_revision(trace, rail="budget_exhausted") is True
    decision = trace["acceptance_decision"]
    assert decision["reason"] == "acceptance_preparation_failed"
    assert "No reviewer rework was requested" in decision["rationale"]
    assert "budget_exhausted" in decision["rationale"]


def test_an_ordinary_dangling_revision_still_terminalizes_on_the_rail(tmp_path):
    from ouroboros.loop_acceptance import terminalize_dangling_revision

    trace = {"acceptance_decision": {"status": "revision_requested", "reason": "improvement_capsule"}}
    assert terminalize_dangling_revision(trace, rail="round_limit") is True
    assert trace["acceptance_decision"]["reason"] == "revision_unavailable_on_forced_rail"


def test_the_incident_rides_the_existing_projections(tmp_path):
    from ouroboros.acceptance_preparation import begin_preparation, incident_projection
    from ouroboros.review_projection import acceptance_decision_projection

    ctx = _ctx(tmp_path)
    record = _fail(begin_preparation(ctx.llm_trace, ctx.tools._ctx))
    projection = acceptance_decision_projection({
        "status": "finalized_unaccepted", "reason": "acceptance_preparation_failed",
        "acceptance_incident": incident_projection(record),
    })
    assert projection["acceptance_incident"]["incident_id"] == record["incident_id"]
    assert projection["acceptance_incident"]["attempts"] == 1
    assert projection["acceptance_incident"]["stage"] == "preparation"


def test_the_checkpoint_publishes_an_incident_with_no_panel_at_all(tmp_path):
    """A local failure produces NO panel; without this the owner's card had
    nothing to read and the warning was invisible."""
    from ouroboros.acceptance_preparation import begin_preparation
    from ouroboros.review_projection import publish_acceptance_checkpoint
    from ouroboros.task_results import load_task_result

    ctx = _ctx(tmp_path)
    record = _fail(begin_preparation(ctx.llm_trace, ctx.tools._ctx))
    publish_acceptance_checkpoint(ctx.tools._ctx, ctx.llm_trace, task_id=ctx.task_id, drive_root=tmp_path)
    result = load_task_result(tmp_path, ctx.task_id) or {}
    incident = (result.get("review_projection") or {}).get("acceptance_incident") or {}
    assert incident["incident_id"] == record["incident_id"]
    assert incident["attempts"] == 1 and incident["status"] == "failed"
    assert (result.get("review_projection") or {}).get("panels") == []


def test_the_projection_merge_keeps_the_incident_of_the_current_publication_only():
    from ouroboros.task_results import merge_review_projection

    panel = {"surface": "task_acceptance", "panel_id": "p1", "publication_revision": 1}
    stale = {"panels": [panel], "acceptance_incident": {"incident_id": "acceptance-preparation:old",
                                                        "status": "failed", "attempts": 1}}
    fresh = {"panels": [{**panel, "publication_revision": 2}]}
    assert "acceptance_incident" not in merge_review_projection(stale, fresh)
    carried = {**fresh, "acceptance_incident": {"incident_id": "acceptance-preparation:new",
                                                "status": "resolved", "attempts": 1}}
    assert merge_review_projection(stale, carried)["acceptance_incident"]["incident_id"] == "acceptance-preparation:new"


def test_the_projection_merge_orders_the_incident_by_publication_in_both_directions():
    """A delayed snapshot or replica can neither erase a fresher warning nor
    resurrect an incident a newer publication resolved — with or without panels."""
    from ouroboros.task_results import merge_review_projection

    failed = {"incident_id": "acceptance-preparation:one", "status": "failed", "attempts": 1}
    resolved = {**failed, "status": "resolved"}
    fresh_warning = {"panels": [], "publication_revision": 3, "acceptance_incident": failed}
    delayed_clean = {"panels": [], "publication_revision": 2}
    merged = merge_review_projection(fresh_warning, delayed_clean)
    assert merged["acceptance_incident"] == failed and merged["publication_revision"] == 3
    assert merge_review_projection(delayed_clean, fresh_warning)["acceptance_incident"] == failed
    later_resolution = {"panels": [], "publication_revision": 4, "acceptance_incident": resolved}
    replayed_failure = {"panels": [], "publication_revision": 3, "acceptance_incident": failed}
    assert merge_review_projection(later_resolution, replayed_failure)["acceptance_incident"] == resolved
    assert merge_review_projection(replayed_failure, later_resolution)["acceptance_incident"] == resolved
    cleared = {"panels": [], "publication_revision": 5}
    assert "acceptance_incident" not in merge_review_projection(later_resolution, cleared)
    assert "acceptance_incident" not in merge_review_projection(cleared, later_resolution)
    assert merge_review_projection(cleared, later_resolution)["publication_revision"] == 5
    # An older snapshot without its own stamp orders by its newest panel.
    panel_only = {"panels": [{"surface": "task_acceptance", "panel_id": "p1", "publication_revision": 2}]}
    assert merge_review_projection(panel_only, fresh_warning)["acceptance_incident"] == failed
    older_incident_only = {"panels": [], "publication_revision": 1, "acceptance_incident": failed}
    merged = merge_review_projection(panel_only, older_incident_only)
    assert "acceptance_incident" not in merged and "publication_revision" not in merged
    assert merged["panels"] == panel_only["panels"]
    # A new task attempt starts its own revisions: its first publication is the newest.
    attempt_one = {"panels": [], "task_attempt": 1, "publication_revision": 7, "acceptance_incident": failed}
    attempt_two = {"panels": [], "task_attempt": 2, "publication_revision": 1}
    assert "acceptance_incident" not in merge_review_projection(attempt_one, attempt_two)
    assert "acceptance_incident" not in merge_review_projection(attempt_two, attempt_one)
    assert merge_review_projection(attempt_two, attempt_one)["task_attempt"] == 2
    # Legacy snapshots without any ordering keep their historical merge: the incoming one wins.
    assert "acceptance_incident" not in merge_review_projection({"panels": [], "acceptance_incident": failed}, {"panels": []})


def test_a_delayed_incident_snapshot_cannot_erase_the_fresher_warning_in_the_task_result(tmp_path):
    from ouroboros.task_results import load_task_result, write_task_result

    failed = {"incident_id": "acceptance-preparation:one", "status": "failed", "attempts": 1}
    write_task_result(tmp_path, "t", "running", review_projection={"panels": [], "publication_revision": 2,
                                                                  "acceptance_incident": failed})
    write_task_result(tmp_path, "t", "running", review_projection={"panels": [], "publication_revision": 1})
    stored = load_task_result(tmp_path, "t")["review_projection"]
    assert stored["acceptance_incident"] == failed and stored["publication_revision"] == 2
    write_task_result(tmp_path, "t", "running", review_projection={"panels": [], "publication_revision": 3})
    stored = load_task_result(tmp_path, "t")["review_projection"]
    assert "acceptance_incident" not in stored and stored["publication_revision"] == 3


def test_artifact_and_receipt_content_change_material_not_delivery_multiplicity(tmp_path):
    from ouroboros.acceptance_preparation import preparation_source_identity
    from ouroboros.outcome_receipt_store import append_verification_receipt, verification_receipts_path

    ctx = _ctx(tmp_path)
    tool_ctx = ctx.tools._ctx
    base = tmp_path / "task_results" / "artifacts" / tool_ctx.task_id
    base.mkdir(parents=True, exist_ok=True)
    output = base / "report.txt"
    output.write_text("a")
    initial = preparation_source_identity(tool_ctx, ctx.llm_trace)
    output.write_text("a")
    assert preparation_source_identity(tool_ctx, ctx.llm_trace) == initial
    output.write_text("b")
    changed = preparation_source_identity(tool_ctx, ctx.llm_trace)
    assert changed["known"] and changed["identity"] != initial["identity"]
    receipt = {"check": "test report", "status": "fail", "ts": "2026-01-01T00:00:00Z"}
    assert append_verification_receipt(tmp_path, tool_ctx.task_id, receipt)
    failed = preparation_source_identity(tool_ctx, ctx.llm_trace)
    assert failed["identity"] != changed["identity"]
    assert append_verification_receipt(tmp_path, tool_ctx.task_id, {**receipt, "ts": "2026-01-01T00:00:01Z"})
    assert preparation_source_identity(tool_ctx, ctx.llm_trace)["identity"] == failed["identity"]
    assert append_verification_receipt(tmp_path, tool_ctx.task_id, {**receipt, "status": "pass", "ts": "2026-01-01T00:00:02Z"})
    assert preparation_source_identity(tool_ctx, ctx.llm_trace)["identity"] != failed["identity"]
    verification_receipts_path(tmp_path, tool_ctx.task_id).write_text("{broken receipt")
    unknown = preparation_source_identity(tool_ctx, ctx.llm_trace)
    assert unknown["identity"] == "unknown" and not unknown["known"]


@pytest.mark.parametrize("action,enforcement", [("finish", "advisory"), ("stop", "blocking")])
def test_an_action_only_informed_author_finish_or_stop_is_honored_with_no_invented_stance(
    tmp_path, monkeypatch, action, enforcement,
):
    """Finding 3: tools/review.py preserves an EMPTY disposition for an action-only
    nomination (C4), but both preparation gates still demanded one of the four
    stance words, so an author who said only finish|stop after an exposed
    preparation failure was refused. Both gates accept the explicit act; the
    recorded author disposition carries the act and no invented stance."""
    from ouroboros import acceptance_preparation as prep, loop as loop_mod
    from ouroboros.loop_acceptance import merge_agent_acceptance_stance
    from ouroboros.loop_delivery import DeliveryCandidate
    from ouroboros.loop_messages import owner_source_sha256

    ctx = _ctx(tmp_path)
    tool_ctx = ctx.tools._ctx
    monkeypatch.setattr(loop_mod, "get_task_review_mode", lambda: "required")
    monkeypatch.setattr(loop_mod, "get_review_enforcement", lambda: enforcement)
    tool_ctx._delivery_candidate = DeliveryCandidate("saved answer", "saved-hash", 1, 1, "prior-fp", {})
    tool_ctx._delivery_candidate.owner_source_sha256 = owner_source_sha256(tool_ctx)
    record = _expose(_fail(prep.begin_preparation(ctx.llm_trace, tool_ctx)))
    merge_agent_acceptance_stance(ctx.llm_trace, {
        "explicit_finish": True, "author_action": action, "disposition": "", "rationale": "the act alone",
    }, tool_ctx)
    assert ctx.llm_trace["acceptance_decision"]["agent_finish_intent"]["author_action"] == action
    assert prep.preparation_delivery_choice(tool_ctx, ctx.llm_trace)
    assert prep.finish_exposed_preparation_author(ctx, record) is True
    decision = ctx.llm_trace["acceptance_decision"]
    assert decision["reason"] == ("author_stop" if action == "stop" else "author_finish")
    assert decision["author_disposition"]["action"] == action
    assert decision["author_disposition"]["disposition"] == ""      # nothing invented
    assert decision["author_disposition"]["subject_hash"] == f"{record['incident_id']}:attempt-1"
