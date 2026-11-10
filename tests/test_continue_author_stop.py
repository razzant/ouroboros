"""The existing Continue over an explicitly unfinished author stop, through the real producer.

``finish_task(action="stop")`` runs through the real loop and the real terminal store; the
row's typed stop then reaches the one offer and admission: a new root on an explicit press,
one successor per predecessor, the same limits, the author's reason as its own account and
nothing started by itself. An ordinary finish, the owner's Stop, Wrap up and Panic, hard
limits, unknown authority and prose stay refused; unsettled writers still hold.
"""

from __future__ import annotations

import base64
import json
import pathlib
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests._store_shared import seed_result
from tests.test_acceptance_author_stop import STOP_RATIONALE, STOP_TEXT, _finish_tool_call, _run_stop_loop, _stop_tool_call

NONCE = "press-0001-abcdef"
GROUP = {"billing_group_id": "stop-root", "billing_group_limit_usd": 20.0,
         "billing_group_limit_source": "initial_task_admission", "billing_group_limit_revision": "admission-1"}
STOP_AXES = {"execution": {"status": "ok", "reason_code": "final_message", "task_completion": {
    "action": "stop", "rationale": STOP_RATIONALE, "answer_sha256": "a" * 64, "source": "finish_task"}},
    "objective": {"status": "fail", "source": "task_completion", "reason": "author_stop"}}


def _stored_author_end(tmp_path, monkeypatch, responses, *, settle=True, ran=None):
    """The real loop's ending, stored by the real terminal writer beside the room's owner words; ``settle``
    runs the real post-task phase (its paid stages stubbed), whose open checkpoint holds any Continue; ``ran``
    (a list) receives the loop's run (its model inputs, trace and usage)."""
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros.owner_mailbox import KIND_OWNER_TEXT, write_owner_message
    from ouroboros.task_results import STATUS_RUNNING, load_task_result, write_task_result

    run = _run_stop_loop(tmp_path, monkeypatch, responses, stop_services=False)
    if ran is not None:
        ran.append(run)
    data = tmp_path / "data"
    write_task_result(data, "stop-root", STATUS_RUNNING, chat_id=7, project_id="", billing_group=GROUP,
                      origin_message_text="Ship the export endpoint.",
                      origin_message_ref={"chat_id": 7, "client_message_id": "m-1"},
                      title="Export endpoint", deadline_at="2099-01-01T00:00:00+00:00",
                      allowed_resources={"network": ["example.invalid"]})
    write_owner_message(data, "Keep the old format too", "stop-root", msg_id="owner-2", kind=KIND_OWNER_TEXT)
    task = {"id": "stop-root", "type": "task", "chat_id": 7, "root_task_id": "stop-root", "delegation_role": "root",
            "text": "Ship the export endpoint."}
    for rel in ("logs", "memory", "repo"):
        (data / rel).mkdir(parents=True, exist_ok=True)
    env = SimpleNamespace(drive_root=data, repo_dir=data / "repo", drive_path=lambda rel: data / rel)
    pipeline._store_task_result(env, task, run.result, run.usage, run.trace)

    def settled():
        from tests.test_agent_task_pipeline import _stubbed_stages

        _stubbed_stages(monkeypatch, [])
        pipeline._run_post_task_processing_async(env, task, run.usage, run.trace, data / "logs", blocking=True)
        return load_task_result(data, "stop-root")

    return data, settled() if settle else settled


def test_a_real_stop_is_offered_and_admits_one_successor_with_the_same_work_and_limits(tmp_path, monkeypatch):
    from ouroboros.main_context_authority import project_helper_predecessor_authority
    from ouroboros.owner_continue import AUTHOR_STOP_CAUSE, continuation_offer
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.continuation_admission import admit_continuation

    data, stored = _stored_author_end(tmp_path, monkeypatch, [_stop_tool_call()])
    from ouroboros.gateway.work_status import shown_outcome
    from ouroboros.project_dialogue import TASK_CAUSE_PHRASES, _completion_verdict, outcome_phase
    from supervisor.events_task_done import _completed_lifecycle_display

    assert stored["status"] == "completed", "the stored status stays the author's own completed row"
    assert stored["outcome_axes"]["execution"]["task_completion"]["rationale"] == STOP_RATIONALE
    assert outcome_phase(stored, {}) == outcome_phase({}, stored) == "cancelled"
    assert shown_outcome(stored) == "cancelled"
    assert _completed_lifecycle_display({}, stored) == ("⏹", "stopped")
    assert _completion_verdict(stored, {}) == TASK_CAUSE_PHRASES["author_stop"][:-1] + " · " + STOP_RATIONALE
    _queue, _state, workers = _install_queue(data, monkeypatch)
    assert continuation_offer(stored, "stop-root") == {"eligible": True, "cause": AUTHOR_STOP_CAUSE, "refusal": ""}
    assert workers.PENDING == [] and "continued_by" not in stored, "the stop and its offer start nothing by themselves"

    ack = admit_continuation("stop-root", action_nonce=NONCE)

    assert ack["ok"] is True and ack["held"] is False and ack["replay"] is False, str(ack)
    successor = ack["successor_task_id"]
    row = next(task for task in workers.PENDING if task["id"] == successor)
    assert row["root_task_id"] == successor and row["chat_id"] == 7 and row["deadline_at"].startswith("2099")
    continuation = row["metadata"]["continuation"]
    assert continuation["cause"] == AUTHOR_STOP_CAUSE and continuation["predecessor_task_id"] == "stop-root"
    assert continuation["work_id"] == "stop-root"
    assert {key: continuation[key] for key in GROUP} == GROUP, "the same bill group and cap, never a new one"
    assert row["metadata"]["task_contract"]["allowed_resources"] == {"network": ["example.invalid"]}
    assert [(r["source"], r["content"]) for r in row["metadata"]["owner_corpus"]] == [
        ("origin_message", "Ship the export endpoint."), ("owner_mailbox", "Keep the old format too")]
    assert "stopped as unfinished" in row["text"] and "was interrupted" not in row["text"]
    assert "recorded cause: author_stop" in row["text"]
    assert f"not an owner instruction; verbatim): {STOP_RATIONALE}" in row["text"]
    assert "Being offered Continue added no retention" in row["text"] and "may already be released" in row["text"], \
        "the gap is disclosed, not hidden"
    assert STOP_RATIONALE not in [r["content"] for r in row["metadata"]["owner_corpus"]], "its account, never owner words"
    assert project_helper_predecessor_authority({"task_id": "stop-root", "status": "completed", "outcome_axes": stored[
        "outcome_axes"]})["outcome_axes"]["execution"]["task_completion"]["rationale"] == STOP_RATIONALE

    # The same press again, on the restored record too: one successor; another press points to it.
    assert admit_continuation("stop-root", action_nonce=NONCE)["successor_task_id"] == successor
    assert [task["id"] for task in workers.PENDING].count(successor) == 1
    workers.PENDING[:] = []
    write_task_result(data, successor, "completed", result="done")
    late = admit_continuation("stop-root", action_nonce=NONCE)
    assert late["replay"] is True and late["successor_task_id"] == successor and workers.PENDING == []
    assert admit_continuation("stop-root", action_nonce="another-press-99") == {
        "ok": False, "error": "already_continued", "successor_task_id": successor}
    claim = load_task_result(data, "stop-root")["continued_by"]
    assert claim["binding"]["cause"] == AUTHOR_STOP_CAUSE and claim["state"] == "admitted"
    assert continuation_offer(load_task_result(data, "stop-root"), "stop-root")["successor_task_id"] == successor


def test_a_real_ordinary_finish_stays_refused_with_the_same_producer(tmp_path, monkeypatch):
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation

    data, stored = _stored_author_end(tmp_path, monkeypatch, [_finish_tool_call("The export endpoint ships.")])
    assert stored["status"] == "completed"
    _queue, _state, workers = _install_queue(data, monkeypatch)
    assert admit_continuation("stop-root", action_nonce=NONCE)["error"] == "author_finished"
    assert workers.PENDING == [] and "continued_by" not in load_task_result(data, "stop-root")


@pytest.mark.parametrize(("reason_code", "axes"), [
    pytest.param("final_message", STOP_AXES, id='final_message-current'),
    pytest.param("final_message", {"execution": {"status": "ok"}, "objective": {"status": "fail", "reason": "review_cycles_exhausted"},
     "review": {"acceptance_decision": {"reason": "review_cycles_exhausted", "author_action": "stop",
                                        "author_disposition": {"action": "stop", "rationale": STOP_RATIONALE}}}}, id='final_message-legacy-exhausted-rounds'),
    pytest.param("plan_review_advisory", {"execution": {"status": "ok"}, "objective": {"status": "fail", "source": "plan_review_author_stop",
                                                  "reason": "author_stop"}}, id='plan_review_advisory-plan-exit'),
    pytest.param("delegated_custody_unreconciled", {"execution": {"status": "ok"}, "objective": {"status": "fail", "source": "task_acceptance_review",
                                                  "reason": "author_stop"},
     "review": {"acceptance_decision": {"reason": "author_stop", "author_action": "stop", "author_disposition": {
         "action": "stop", "rationale": STOP_RATIONALE}}}}, id='delegated_custody_unreconciled-legacy-decision'),
])
def test_every_stored_stop_shape_reads_the_same_under_its_recorded_reason(axes, reason_code):
    from ouroboros.owner_continue import AUTHOR_STOP_CAUSE, continuation_eligibility

    verdict = continuation_eligibility({"status": "completed", "reason_code": reason_code, "root_task_id": "r",
                                        "outcome_axes": {**axes, "execution": {**axes["execution"],
                                                                               "reason_code": "final_message"}}}, "r")
    assert verdict == {"eligible": True, "cause": AUTHOR_STOP_CAUSE, "refusal": ""}


def _row(**fields):
    return {"status": "completed", "reason_code": "final_message", "root_task_id": "r", "outcome_axes": STOP_AXES, **fields}


@pytest.mark.parametrize("row,refusal", [
    ({"status": "completed", "reason_code": "final_message", "root_task_id": "r"}, "author_finished"),
    (_row(status="failed", cancel_origin={"source": "http_single"}), "stopped_by_owner"),
    (_row(deadline_at="not-a-time"), "deadline_unknown"),
], ids=["plain-finish", "owner-stop",
        "deadline-unknown"])
def test_only_a_typed_author_stop_opens_continue(row, refusal):
    from ouroboros.model_wait import EXECUTION_CLOSE_KEY
    from ouroboros.owner_continue import continuation_eligibility

    if "execution_close" in row:
        row = {**row, EXECUTION_CLOSE_KEY: row["execution_close"]}
    verdict = continuation_eligibility(row, "r")
    assert verdict["eligible"] is False and verdict["refusal"] == refusal, verdict


def test_a_stop_at_a_technical_rail_keeps_its_own_cause():
    from ouroboros.owner_continue import continuation_eligibility

    assert continuation_eligibility(_row(reason_code="round_limit"), "r")["cause"] == "round_limit"


def test_a_stop_with_a_finite_budget_carries_the_closed_time(tmp_path, monkeypatch):
    from ouroboros.model_wait import EXECUTION_CLOSE_KEY
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation

    data, stored = _stored_author_end(tmp_path, monkeypatch, [_stop_tool_call()])
    seed_result(data, "stop-root", {**stored, "execution_timeout_sec": 600, "task_attempt": 1,
                                    EXECUTION_CLOSE_KEY: {"sec": 125.5, "attempt": 1}})
    _queue, _state, workers = _install_queue(data, monkeypatch)
    ack = admit_continuation("stop-root", action_nonce=NONCE)
    assert ack["ok"] is True, ack
    row = next(task for task in workers.PENDING if task["id"] == ack["successor_task_id"])
    assert row["execution_timeout_sec"] == 600 and row["_execution_handoff_sec"] == 125.5
    assert load_task_result(data, "stop-root")["continued_by"]["binding"]["execution_consumed_sec"] == 125.5


def test_its_open_post_task_phase_holds_the_successor_until_that_phase_settles(tmp_path, monkeypatch):
    """The producer's own late phase is a writer: the press is admitted held, and the phase's end releases it."""
    from supervisor.continuation_admission import admit_continuation, release_settled_continuations
    from supervisor.events_budget import budget_hold_fact

    data, settle = _stored_author_end(tmp_path, monkeypatch, [_stop_tool_call()], settle=False)
    _queue, _state, workers = _install_queue(data, monkeypatch)
    ack = admit_continuation("stop-root", action_nonce=NONCE)
    assert ack["ok"] is True and ack["held"] is True
    assert ack["blockers"] == [{"kind": "late_phase_running", "task_id": "stop-root"}]
    assert release_settled_continuations("stop-root") == [], "no timer or second writer while it runs"
    held = next(row for row in workers.PENDING if row["id"] == ack["successor_task_id"])
    assert budget_hold_fact(held)["reason"] == "continuation_writer_unsettled"
    settle()  # the phase's own end reconciles it: no second press, and the row stays the one successor
    assert budget_hold_fact(held) is None and [row["id"] for row in workers.PENDING] == [ack["successor_task_id"]]
    assert admit_continuation("stop-root", action_nonce=NONCE)["held"] is False


@pytest.mark.parametrize("spent,admitted", [(19.0, True), (20.0, False), (25.0, False)])
def test_spent_group_money_refuses_the_press_with_nothing_written(tmp_path, monkeypatch, spent, admitted):
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation
    from tests.test_billing_group import _scope, _spend

    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    data, _stored = _stored_author_end(tmp_path, monkeypatch, [_stop_tool_call()])
    _queue, _state, workers = _install_queue(data, monkeypatch)
    _spend(data, _scope(data, "stop-root", "stop-root", group="stop-root", group_limit=20.0, root_limit=20.0), spent)
    ack = admit_continuation("stop-root", action_nonce=NONCE)
    assert ack["ok"] is admitted, str(ack)
    if admitted:
        assert load_task_result(data, "stop-root")["continued_by"]["binding"]["billing_group_limit_usd"] == 20.0
        return
    assert ack == {"ok": False, "error": "hard_limit_reached", "cause": "budget_exhausted"}
    assert workers.PENDING == [] and "continued_by" not in load_task_result(data, "stop-root")


def test_an_unreadable_ledger_refuses_the_press_as_unknown_never_as_money(tmp_path, monkeypatch):
    from ouroboros import usage_accounting
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation

    data, _stored = _stored_author_end(tmp_path, monkeypatch, [_stop_tool_call()])
    _queue, _state, workers = _install_queue(data, monkeypatch)

    def unreadable(*_args, **_kwargs):
        raise OSError("ledger unreadable")

    monkeypatch.setattr(usage_accounting, "usage_projection", unreadable)
    assert admit_continuation("stop-root", action_nonce=NONCE) == {"ok": False, "error": "billing_authority_unavailable"}
    assert workers.PENDING == [] and "continued_by" not in load_task_result(data, "stop-root")


def test_a_stop_cancels_no_child_so_a_running_child_holds_the_one_successor(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from supervisor.continuation_admission import admit_continuation

    data, _stored = _stored_author_end(tmp_path, monkeypatch, [_stop_tool_call()])
    _queue, _state, workers = _install_queue(data, monkeypatch)
    write_task_result(data, "stop-child", STATUS_RUNNING, chat_id=7)
    workers.RUNNING["stop-child"] = {"task": {"id": "stop-child", "root_task_id": "stop-root",
                                              "parent_task_id": "stop-root"}, "worker_id": 0, "attempt": 1}
    ack = admit_continuation("stop-root", action_nonce=NONCE)
    assert ack["ok"] is True and ack["held"] is True
    assert ack["blockers"] == [{"kind": "running_member", "task_id": "stop-child"}]
    assert [task["id"] for task in workers.PENDING] == [ack["successor_task_id"]]


def test_the_successor_starts_with_the_stop_in_its_brief(tmp_path, monkeypatch):
    from ouroboros.agent_startup_checks import validate_task_authority_sources
    from supervisor.continuation_admission import admit_continuation

    data, _stored = _stored_author_end(tmp_path, monkeypatch, [_stop_tool_call()])
    _queue, _state, workers = _install_queue(data, monkeypatch)
    ack = admit_continuation("stop-root", action_nonce=NONCE)
    task = dict(next(row for row in workers.PENDING if row["id"] == ack["successor_task_id"]))
    env = SimpleNamespace(drive_root=data, budget_drive_root=data)
    assert validate_task_authority_sources(env, task) == {}
    brief = task["predecessor_authority"]
    assert brief["status"] == "completed" and brief["source"]["task_id"] == "stop-root"
    assert brief["outcome_axes"]["execution"]["task_completion"]["rationale"] == STOP_RATIONALE
    assert STOP_TEXT not in str(brief.get("result") or ""), "the answer stays behind its exact reader"


def _line(task_id):
    return SimpleNamespace(review_owned=False, run_id=f"run-of-{task_id}", task_id=task_id, root_task_id=task_id,
                           invocation_id="")


def test_a_stop_offer_holds_no_resources_while_an_interruption_still_does(tmp_path, monkeypatch):
    """The offer and the resources' lifetime stay apart: a stop keeps a finished root's ordinary retention, and an
    admitted successor's live line holds what is left, as for any Continue."""
    from ouroboros.delegate_continuation import still_continuable
    from ouroboros.owner_continue import continuation_offer
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.continuation_admission import admit_continuation

    data, stored = _stored_author_end(tmp_path, monkeypatch, [_stop_tool_call()])
    _queue, _state, workers = _install_queue(data, monkeypatch)
    assert continuation_offer(stored, "stop-root")["eligible"] is True
    assert still_continuable(data, _line("stop-root"), set()) is False
    assert still_continuable(data, _line("stop-root"), set(), every_offer=True) is True, "a managed document's release"
    write_task_result(data, "broken-root", "failed", chat_id=7, root_task_id="broken-root", reason_code="provider_failure")
    assert still_continuable(data, _line("broken-root"), set()) is True, "an interruption's hold is unchanged"
    write_task_result(data, "unknown-root", "failed", chat_id=7, root_task_id="unknown-root")  # no cause recorded
    assert continuation_offer(load_task_result(data, "unknown-root"), "unknown-root")["refusal"] == (
        "interruption_cause_unrecorded")
    assert still_continuable(data, _line("unknown-root"), set()) is False, "the registration reading is unchanged"
    assert still_continuable(data, _line("unknown-root"), set(), every_offer=True) is True, "undecided for documents"
    ack = admit_continuation("stop-root", action_nonce=NONCE)
    assert load_task_result(data, "stop-root")["continued_by"]["successor_task_id"] == ack["successor_task_id"]
    assert still_continuable(data, _line("stop-root"), {ack["successor_task_id"]}) is True
    assert [row["id"] for row in workers.PENDING] == [ack["successor_task_id"]]


def test_the_incident_replayed_as_facts_a_stop_over_running_work_reaches_the_next_turn_and_its_continue(
        tmp_path, monkeypatch):
    """10.11: a root left its change review for an addressed message, asked a plan seat, and at a full window called
    finish_task(action="stop", pending_review="wait"). It ended in that batch, as a stop does; the owner's next turn
    asked why and answered from `status: completed`. Replayed as facts (the review and the plan seat are their tools'
    own returns): the stop ends in its batch, its receipt says the argument had no effect, its summary is kept; the
    next owner-started turn's metadata names the stopped work, its card word, its own reason as its author's account
    and its open Continue; continue_task from that turn admits one successor and a second request names the same one.
    Nothing here asserts what a model then does."""
    import queue

    from ouroboros import projects_registry
    from ouroboros.review_pause import PROCESS_TOKEN
    from ouroboros.server_routing_context import STOP_REASON_LABEL, _decision_turn_metadata
    from ouroboros.tool_call_log import invocation_scope
    from ouroboros.tools import plan_review, review_change
    from ouroboros.tools.control_mind import _continue_task
    from supervisor.events_mind_controls import _apply

    def leave_the_wave(ctx, review_reference=None, *, continue_seats=None, _resolved_binding=None, **_args):
        ctx._left_review_wait = {"surface": "change", "binding": "change-review:k2", "retry_key": "k2",
                                 "owner": PROCESS_TOKEN, "done": False, "msg_id": "change-review-settled:k2",
                                 "record_id": "rl-review-2"}
        return ("An addressed message reached you while the reviewers ran: this call returned and their paid work "
                "continues separately.\n\n" + json.dumps({"record_id": "rl-review-2", "state": "pending"}))

    monkeypatch.setattr(review_change, "_handle_review_change", leave_the_wave)
    monkeypatch.setattr(plan_review, "_handle_plan_task", lambda _ctx, **_kw: "Ask a6 sent. Still working: seat-a.")
    summary = "Stopped at an honest midpoint: fixes ready, the fresh review not finished."

    def call(name, arguments, call_id):
        return {"content": None, "tool_calls": [{"id": call_id, "type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments)}}]}

    def stop():
        return call("finish_task", {"action": "stop", "answer": STOP_TEXT, "rationale": STOP_RATIONALE,
                                    "pending_review": "wait", "summary": summary}, "stop-1")

    runs = []
    data, stored = _stored_author_end(tmp_path, monkeypatch, [
        call("review_change", {"subject": "index"}, "review-2"),
        call("plan_task", {"ask": ["seat-a"], "message": "Is the amended candidate safe?"}, "ask-a6"),
        stop()], ran=runs)
    [run] = runs
    assert [call["tool"] for call in run.trace["tool_calls"]] == ["review_change", "plan_task", "finish_task"]
    assert len(run.model_inputs) == 3, "the stop ended in its batch: no round after it"
    receipt = json.loads(run.trace["tool_calls"][-1]["result"])
    assert "no effect on a stop" in receipt["pending_review"]
    assert run.usage["wake_summary"] == summary and stored["wake_summary"] == summary
    assert stored["outcome_axes"]["execution"]["task_completion"]["rationale"] == STOP_RATIONALE

    projects_registry.create_project(data, "elsewhere", name="Elsewhere")  # Main lists its rooms beside results
    ctx = SimpleNamespace(DRIVE_ROOT=data, PENDING=[], RUNNING={}, load_state=lambda: {"owner_id": 1, "owner_chat_id": 1},
                          update_state=lambda fn: fn({"owner_id": 1, "owner_chat_id": 1}))
    metadata = _decision_turn_metadata(ctx, 7, "cm-why-stopped", {})  # «почему остановился? доделали?»
    reference = next(row for row in metadata["main_routing_manifest"]["final_results"] if row["task_id"] == "stop-root")
    assert (reference["status"], reference["outcome"]) == ("completed", "Stopped")
    assert reference["continuation_offer"] == {"eligible": True, "cause": "author_stop", "refusal": ""}
    assert reference["stop_reason"] == {"verbatim": STOP_RATIONALE, "label": STOP_REASON_LABEL}
    assert "continue_task" in metadata["routing_contract"]["valid_actions"]

    _queue, _state, workers = _install_queue(data, monkeypatch)
    turn = SimpleNamespace(task_id="why-turn", drive_root=data, event_queue=queue.Queue())
    outcomes = []
    for call_id in ("continue-1", "continue-2"):
        with invocation_scope({"tool_call_id": call_id}):
            assert json.loads(_continue_task(turn, "stop-root", "My human asked me to go on."))["status"] == "requested"
        outcomes.append(_apply(turn.event_queue.get_nowait()))
    first, second = outcomes
    assert first["ok"] is True and [task["id"] for task in workers.PENDING] == [first["successor_task_id"]]
    assert second == {"ok": False, "error": "already_continued", "successor_task_id": first["successor_task_id"]}
    again = _decision_turn_metadata(ctx, 7, "cm-after", {})
    reference = next(row for row in again["main_routing_manifest"]["final_results"] if row["task_id"] == "stop-root")
    assert reference["continuation_offer"]["successor_task_id"] == first["successor_task_id"]
    assert reference["continuation_offer"]["eligible"] is False


def test_the_finish_task_schema_names_the_stop_continue_boundary(tmp_path):
    from ouroboros.tools.registry import ToolRegistry

    schema = next(row for row in ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path).schemas()
                  if (row.get("function") or row).get("name") == "finish_task")
    text = (schema.get("function") or schema)["description"]
    assert "A root's stop records unfinished work for explicit Continue" in text and "under the same limits" in text
    assert "Continue does not reopen an ordinary finish or a final reply without this tool" in text
    assert "even when its text calls it partial" in text
    # Finish first; a stop for a step still mine names the rails that can end a turn, and a full window is not one
    # of them while the host can renew this task's view (``context_renewal``), which can still refuse.
    for fact in ("Use finish when nothing of this work is left for you", "a deadline, a budget or a round limit",
                 "A full window is not such a limit", "after a real size refusal",
                 "an irreducible or non-shrinking view", "a directly required handbook"):
        assert fact in text, fact
    pending = (schema.get("function") or schema)["parameters"]["properties"]["pending_review"]["description"]
    # A stop waits for nothing; the act that waits is the review's own collection, live here, recorded after a restart
    # (``review_change_custody.collect_recorded_commit_actor``'s cold path never blocks).
    for fact in ("A stop waits for nothing", "collect it with its review_reference", "still running in this process",
                 "after a restart", "possibly still pending"):
        assert fact in pending, fact


def test_the_tool_description_says_what_the_door_admits():
    from ouroboros.tools.control_mind import mind_tool_entries

    entry = next(row for row in mind_tool_entries() if row.name == "continue_task")
    text = entry.schema["description"]
    assert "finish_task action=stop" in text and "ordinary finished answer" in text and "owner Stop/Panic" in text


# --- my human's image message that the loop read before its author stopped (#2941) ---------------------

WORDS = "Keep this exact wording.\n[image: literal owner syntax]\nDo not publish yet."  # image-like words are words
MID = "client-image-1:stop-root"  # the door's id for this message in this task
REPLY = {"address": "row:7@2026-11-07T21:59:00+00:00#owner-earlier", "unresolved": True}
FILES = {"one_image": ("a.png",), "two_images": ("a.png", "b.png"), "manifest_ref": ("notes.txt", "a.png"),
         "pixels_unavailable": ("a.png",), "not_composed": ("a.png",)}


def _continue_door():
    """The real HTTP Continue door (``gateway/task_continue.py``) on its own route."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.task_continue import api_task_continue

    return TestClient(Starlette(routes=[Route("/api/tasks/{task_id}/continue", api_task_continue, methods=["POST"])]))


def _image_author_end(tmp_path, monkeypatch, variant, tamper=None):
    """The server door writes my human's image message before round one; the real loop reads it, its author stops,
    and the real terminal collector, store and post-task phase settle the row. ``tamper`` edits the recorded
    directive before the terminal captures it (a genuinely conflicting record)."""
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros import artifacts, loop_round_limits
    from ouroboros.task_results import STATUS_RUNNING, load_task_result, write_task_result
    from tests._owner_image_mail import png_bytes, write_owner_images
    from tests.test_agent_task_pipeline import _stubbed_stages

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    for index, name in enumerate(FILES[variant]):
        (uploads / name).write_bytes(png_bytes(200 + index) if name.endswith(".png") else b"owner notes\n")
    if variant == "manifest_ref":  # the image is only in the complete manifest, past the row's inline view
        monkeypatch.setattr(artifacts, "_MAX_STAGED_ATTACHMENTS", 1)
    staged = {}

    def owner_mail(data):
        staged["text"], staged["manifest"] = write_owner_images(
            data, "stop-root", WORDS, [uploads / name for name in FILES[variant]], msg_id=MID,
            client_message_id="client-image-1", reply_reference=REPLY)
        if variant == "pixels_unavailable":  # the staged bytes change after their identity was captured
            image = artifacts.task_artifact_dir_path(data, "stop-root") / staged["manifest"][0]["relpath"]
            image.write_bytes(b"changed after staging")
        if variant == "not_composed":
            def broken(*_args, **_kwargs):
                raise RuntimeError("decoder crashed")

            monkeypatch.setattr(loop_round_limits, "build_incoming_user_content", broken)

    run = _run_stop_loop(tmp_path, monkeypatch, [_stop_tool_call()], stop_services=False, before_loop=owner_mail)
    assert run.trace["task_completion"]["action"] == "stop" and len(run.model_inputs) == 1
    data = tmp_path / "data"
    write_task_result(data, "stop-root", STATUS_RUNNING, chat_id=7, project_id="", billing_group=GROUP,
                      origin_message_text="Ship the export endpoint.",
                      origin_message_ref={"chat_id": 7, "client_message_id": "m-1"},
                      title="Export endpoint", deadline_at="2099-01-01T00:00:00+00:00")
    task = {"id": "stop-root", "type": "task", "chat_id": 7, "root_task_id": "stop-root", "delegation_role": "root",
            "text": "Ship the export endpoint."}
    for rel in ("logs", "memory", "repo"):
        (data / rel).mkdir(parents=True, exist_ok=True)
    env = SimpleNamespace(drive_root=data, repo_dir=data / "repo", drive_path=lambda rel: data / rel)
    if tamper is not None:
        tamper(next(row for row in run.ctx._owner_directives if row.get("msg_id") == MID), staged["text"])
    evidence = pipeline._collect_review_evidence_bundle(env, run.ctx, task, data, [])
    pipeline._store_task_result(env, task, run.result, run.usage, run.trace, review_evidence=evidence)
    _stubbed_stages(monkeypatch, [])
    pipeline._run_post_task_processing_async(env, task, run.usage, run.trace, data / "logs", blocking=True)
    stored = load_task_result(data, "stop-root")
    assert stored["status"] == "completed", stored.get("status")
    return data, stored, staged, run


def _own_images(data, row):
    """(sha256, bytes in ROW's OWN store) of each staged image ROW declares."""
    from ouroboros.artifacts import task_artifact_dir_path

    store = task_artifact_dir_path(pathlib.Path(row.get("drive_root") or data), row["id"])
    return [(image["sha256"], (store / image["relpath"]).read_bytes()) for image in row.get("attachment_images") or []]


@pytest.mark.parametrize("variant", list(FILES))
def test_an_owner_image_the_loop_read_continues_with_its_exact_words_and_files(tmp_path, monkeypatch, variant):
    """#2941: a message of my human's with an image, read by the real loop before its author stopped, was refused
    as ``owner_message_identity_conflict``: the terminal corpus kept the drain's rendering (the words, a generated
    caption, an image reference) under the same id as the captured mailbox row's exact words. The recorded row now
    keeps those ``words`` beside its rendering, so the HTTP door admits one successor whose corpus holds the exact
    words once, under their id and reply target; each image is copied into the successor's own store and read
    natively by its first request, and a second Continue still holds it after the first task's copies are gone. A
    rendering error or a failed composition changes no words; bytes that changed stay a named gap."""
    import hashlib
    import shutil

    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.context import _first_user_content
    from ouroboros.owner_mailbox import _mailbox_path
    from ouroboros.task_custody import settle_task_mailbox
    from ouroboros.task_results import load_task_result, write_task_result

    data, stored, staged, run = _image_author_end(tmp_path, monkeypatch, variant)
    text = staged["text"]
    pixels = [(tmp_path / "uploads" / name).read_bytes() for name in FILES[variant] if name.endswith(".png")]
    assert [json.loads(line)["text"] for line in stored["owner_mailbox"]["rows"]
            if json.loads(line)["msg_id"] == MID] == [text]
    if variant != "two_images":  # the settled mailbox goes: Continue reads the captured row alone
        assert settle_task_mailbox(data, "stop-root", data) and not _mailbox_path(data, "stop-root").exists()
    _queue, _state, workers = _install_queue(data, monkeypatch)
    door = _continue_door()

    first = door.post("/api/tasks/stop-root/continue", json={"action_nonce": NONCE})

    assert first.status_code == 200 and first.json()["ok"] is True, first.text
    [directive] = [row for row in run.ctx._owner_directives if row.get("msg_id") == MID]
    [corpus] = [row for row in stored["review_evidence"]["task_inputs"]["owner_requirements_and_decisions"]
                if row.get("msg_id") == MID]
    assert directive["words"] == corpus["words"] == text and corpus["content"] != text, "the rendering stays beside"
    if variant in {"one_image", "two_images", "manifest_ref"}:
        assert corpus["content"].count("[image ref sha256:") == len(pixels), "reviewers keep the image references"
        sent = [part for message in run.model_inputs[0] if isinstance(message.get("content"), list)
                for part in message["content"] if part.get("type") == "image_url"]
        assert [base64.b64decode(part["image_url"]["url"].split(",", 1)[1]) for part in sent] == pixels
    successor = next(row for row in workers.PENDING if row["id"] == first.json()["successor_task_id"])
    mine = [row for row in successor["metadata"]["owner_corpus"] if row.get("msg_id") == MID]
    assert mine == [{"source": "owner_mailbox", "content": text, "msg_id": MID,
                     "reply_to": {"address": REPLY["address"], "author": "", "text": ""}}]
    assert successor["text"].count(WORDS) == 1, "the exact words once, never beside a rendering of them"
    again = door.post("/api/tasks/stop-root/continue", json={"action_nonce": NONCE})
    assert again.status_code == 200 and again.json()["successor_task_id"] == successor["id"]
    assert [row["id"] for row in workers.PENDING].count(successor["id"]) == 1, "the same press, one successor"
    originals = [hashlib.sha256(raw).hexdigest() for raw in pixels]
    if variant == "pixels_unavailable":
        assert _own_images(data, successor) == []
        assert [(row["status"], row["reason"], row["sha256"]) for row in successor["attachments"]] == [
            ("rejected", "source_missing", originals[0])], "changed bytes are a named gap, never other pixels"
    else:
        assert _own_images(data, successor) == list(zip(originals, pixels)), "verified copies in its own store"
        first_request = _first_user_content(successor, SimpleNamespace(drive_root=data))
        native = [part for part in first_request if part.get("type") == "image_url"]
        assert [base64.b64decode(part["image_url"]["url"].split(",", 1)[1]) for part in native] == pixels
        own = task_artifact_dir_path(data, successor["id"]).resolve()
        assert all(pathlib.Path(part["_source_path"]).is_relative_to(own) for part in native)

    shutil.rmtree(task_artifact_dir_path(data, "stop-root") / "attachments")  # the first task's copies are gone
    workers.PENDING[:] = []
    write_task_result(data, successor["id"], "cancelled", result="interrupted",
                      cancel_origin={"source": "snapshot_restore", "reason": "server_shutdown"})
    second = door.post(f"/api/tasks/{successor['id']}/continue", json={"action_nonce": "second-press-2941"})

    assert second.status_code == 200 and second.json()["ok"] is True, second.text
    hop = next(row for row in workers.PENDING if row["id"] == second.json()["successor_task_id"])
    assert [row for row in hop["metadata"]["owner_corpus"] if row.get("msg_id") == MID] == mine
    assert _own_images(data, hop) == ([] if variant == "pixels_unavailable" else list(zip(originals, pixels)))
    assert "continued_by" in load_task_result(data, successor["id"])


@pytest.mark.parametrize("record, gap", [
    ("rival_words", "owner_message_identity_conflict"),
    ("peer_label", "owner_message_identity_conflict"),
    ("malformed_words", "retained_owner_corpus_row_malformed"),
])
def test_a_genuine_rival_of_an_image_message_still_refuses_continue(tmp_path, monkeypatch, record, gap):
    """The words beside a rendering exempt nothing: other words under the image message's id, the same words
    labelled a task's (peer) message, or ``words`` that are not text refuse at the door before any claim or row."""
    from ouroboros.task_results import load_task_result

    def tamper(row, text):
        if record == "rival_words":
            row["words"] = text + "\nActually publish it now."
        elif record == "peer_label":
            row.update(source="principal_task_message", source_task_id="peer-task")
        else:
            row["words"] = ["not", "text"]

    data, _stored, _staged, _run = _image_author_end(tmp_path, monkeypatch, "one_image", tamper=tamper)
    _queue, _state, workers = _install_queue(data, monkeypatch)

    refused = _continue_door().post("/api/tasks/stop-root/continue", json={"action_nonce": NONCE})

    assert refused.status_code == 409, refused.text
    assert refused.json()["reason_code"] == "owner_source_missing" and refused.json()["gaps"] == [gap]
    assert workers.PENDING == [] and "continued_by" not in load_task_result(data, "stop-root")
