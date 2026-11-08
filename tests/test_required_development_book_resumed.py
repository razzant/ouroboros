"""A start refused while preparing its context, on the real lifecycle (the pair of the required handbook).

Its context capture refuses it: a known window that cannot hold the required handbook (``context_capacity``), or that
handbook unreadable (``required_reference_unavailable``, the producer's own cause under the existing source-unavailable
outcome). A resumed attempt: attempt 2's own loop writer saves work and a run starts, its worker dies, the
supervisor's retry freezes that work for attempt 3 and the queue's admission check accepts the locator. The owner's
Continue: a root ``start_task`` admitted with the requirement ends refused, and only the existing Continue continues it.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests._store_shared import seed_result
from tests.test_owner_continue import NONCE
from tests.test_required_development_book import (
    KEY,
    _agent_world,
    _isolated_projects_root,  # noqa: F401 -- the autouse isolation fixture rides the import
    _queued,
    _real_agent,
)
from tests.test_root_effort_ingress import _promoted

CAUSES = ("context_capacity", "required_reference_unavailable")
_UNKNOWN_MONEY = {"cost_accounting_status": "ok", "cost_final": False, "accounted_upper_bound_usd": None,
                  "total_rounds": None, "prompt_tokens": None, "completion_tokens": None, "reserved_usd": 1.5,
                  "unresolved_upper_bound_usd": 1.5, "unknown_unmetered": True, "cost_presentation": None}


def _refusing_world(tmp_path, monkeypatch, cause):
    from ouroboros import agent_task_pipeline as pipeline

    capacity = cause == "context_capacity"
    repo, drive, ran = _agent_world(tmp_path, monkeypatch, 150_000 if capacity else 1_000_000)
    if not capacity:
        (repo / "docs/development/change.md").unlink()
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **kw: pytest.fail("no paid synthesis"))
    monkeypatch.setattr("supervisor.state.reconstruct_task_cost", lambda *_a, **_kw: dict(_UNKNOWN_MONEY))
    return repo, drive, ran


def _resumed_after_a_crash(drive, prior):
    from ouroboros import delegate_custody as custody
    from ouroboros import working_checkpoint
    from ouroboros.delegate_recovery import prepare_worker_crash_handoff
    from ouroboros.task_results import STATUS_INTERRUPTED, write_task_result

    task_id = prior["id"]
    write_task_result(drive, task_id, "running", delegated_runs_unreconciled=["run-a1"], root_task_id=task_id,
                      task_attempt=2)
    assert custody.emit(drive, custody.STARTED, {"task_id": task_id, "run_id": "run-a1"})
    ctx = SimpleNamespace(task_id=task_id, task_attempt=2, task_started_at=1_700_000_000.0, drive_root=drive,
                          budget_drive_root=drive)
    call = {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
    work = SimpleNamespace(
        tools=SimpleNamespace(_ctx=ctx), round_idx=3, tool_schemas=[], owner_msg_seen=set(), budget_tail="tool",
        messages=[{"role": "user", "content": "Work"}, {"role": "assistant", "content": "", "tool_calls": [call]},
                  {"role": "tool", "tool_call_id": "c1", "content": "read"}],
        llm_trace={"reasoning_notes": [], "tool_calls": [{"tool": "read_file", "args": {}, "result": "read"}]},
        accumulated_usage={"rounds": 3, "prompt_tokens": 1_200, "completion_tokens": 300, "cost": 0.42})
    assert working_checkpoint.save_round(work, "post_batch")
    saved = working_checkpoint.read_working(drive, task_id, 2)[0]  # a recovery freezes the whole state
    task = {**prior, "_attempt": 3}
    assert working_checkpoint.attach_recovery(drive, task, source_task_id=task_id, from_attempt=2, cause="worker_crash",
                                              prior_task=prior)
    prepare_worker_crash_handoff(drive, task, old_attempt=2, new_attempt=3, worker_id=1, exitcode=-9)
    write_task_result(drive, task_id, STATUS_INTERRUPTED, task_attempt=2,
                      result="Worker process died mid-task (attempt 2). Retrying.")
    assert working_checkpoint.recovery_source_for_task(drive, task)["usage"]["rounds"] == 3
    return task, dict(task["_working_recovery"]), saved


@pytest.mark.parametrize("cause", CAUSES)
def test_a_resumed_attempt_refused_at_its_start_states_no_tally_and_its_frozen_work_stays(tmp_path, monkeypatch, cause):
    """A same-ID resume carries the cumulative tally, so a start its own capture refused states none: it never says
    that no rounds ran. Each refusal keeps its own cause. Money stays the ledger's, the started run stays disclosed,
    and the saved work stays in its frozen custody: it still loads after the terminal, also where the durable result
    retired the rolling duplicate."""
    from ouroboros import working_checkpoint
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.task_results import load_task_result

    repo, drive, ran = _refusing_world(tmp_path, monkeypatch, cause)
    task, locator, saved = _resumed_after_a_crash(drive, _queued("resumed", _attempt=2, task_contract={KEY: True}))
    events = _real_agent(repo, drive)._handle_task_scoped(task)
    done = next(row for row in events if row["type"] == "task_done")
    stored = load_task_result(drive, "resumed")
    assert ran == [] and done["reason_code"] == cause and stored["outcome_axes"]["execution"]["reason_code"] == cause
    assert stored["task_attempt"] == 3
    assert stored["outcome_axes"]["execution"]["failure"] == {"kind": "runtime", "reason_code": cause}
    assert done["delegated_runs_unreconciled"] == stored["delegated_runs_unreconciled"] == ["run-a1"]
    assert done["unknown_unmetered"] is True and done["unresolved_upper_bound_usd"] == 1.5  # the ledger's answer
    tally = stored["loop_outcome"]["usage"]
    assert (tally["total_rounds"], tally["prompt_tokens"], tally["completion_tokens"]) == (None, None, None)
    assert next(row for row in events if row["type"] == "task_metrics")["rounds"] is None
    assert "call count unknown" in stored["trace_summary"]
    assert read_actor_source_bytes(drive, "resumed", locator["source_ref"]) == saved
    reader = SimpleNamespace(task_id="resumed", drive_root=drive, budget_drive_root=drive, task_metadata={})
    frozen = working_checkpoint.load_recovery(reader, locator)
    assert frozen["usage"]["rounds"] == 3 and frozen["messages"][-1]["content"] == "read"
    # The capacity terminal leaves the rolling duplicate; for the source cause the durable result retires it.
    assert working_checkpoint.checkpoint_path(drive, "resumed", 2).exists() is (cause == "context_capacity")


# --- The owner's Continue of a refused start --------------------------------------------------------------------------

def _refused_root(tmp_path, monkeypatch, cause, *, resumed=False):
    """A root ``start_task(context_requires_development=true)`` through its real admission, dispatched and ended by
    its own capture's refusal in the real agent; the queue holds nothing of it any more."""
    from ouroboros.task_results import load_task_result

    admitted, _row, workers = _promoted(tmp_path, monkeypatch, **{KEY: True})
    workers.PENDING[:] = [task for task in workers.PENDING if task["id"] != admitted["id"]]
    repo, drive, ran = _refusing_world(tmp_path, monkeypatch, cause)
    task, frozen = {**admitted, "drive_root": str(drive)}, None
    if resumed:
        task, locator, saved = _resumed_after_a_crash(drive, {**task, "_attempt": 2})
        frozen = (locator, saved)
    _real_agent(repo, drive)._handle_task_scoped(task)
    stored = load_task_result(drive, task["id"])
    assert ran == [] and stored["status"] == "failed" and stored["reason_code"] == cause
    assert stored["task_contract"][KEY] is True
    return task["id"], stored, workers, frozen


@pytest.mark.parametrize("cause", CAUSES)
def test_only_the_owners_continue_starts_the_same_work_again_with_its_requirement_history_and_money(
        tmp_path, monkeypatch, cause):
    """Nothing starts again by itself. The card offers the ordinary Continue, which admits a new root keeping the
    exact own requirement, the owner's original words, the predecessor's record as its source and the work's money
    group; it neither resumes the old task nor changes its route or requirement."""
    from ouroboros.context_input_selection import development_required
    from ouroboros.owner_continue import continuation_offer
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation

    task_id, stored, workers, _frozen = _refused_root(tmp_path, monkeypatch, cause)
    assert workers.PENDING == [] and "continued_by" not in stored  # no retry, no successor of its own accord
    assert continuation_offer(stored, task_id) == {"eligible": True, "cause": cause, "refusal": ""}
    ack = admit_continuation(task_id, action_nonce=NONCE)
    assert ack["ok"] is True and ack["held"] is False, ack
    [row] = [task for task in workers.PENDING if task["id"] == ack["successor_task_id"]]
    assert row["task_contract"][KEY] is True and KEY not in row and development_required(row) is True
    assert not {"model", "use_local_model", "_working_recovery"} & set(row)  # no route switched, nothing resumed
    continuation = row["metadata"]["continuation"]
    assert continuation["predecessor_task_id"] == task_id and continuation["cause"] == cause
    assert [source["content"] for source in row["metadata"]["owner_corpus"]] == ["please research this properly"]
    assert f"Continue was requested for task {task_id}" in row["text"] and "the old task is not resumed" in row["text"]
    assert {key: row["predecessor_authority_source"][key] for key in ("kind", "task_id", "tool")} == {
        "kind": "task_result", "task_id": task_id, "tool": "get_task_result"}  # its record is the successor's source
    admitted_under = stored["metadata"]["billing_group"]  # the group and cap its own admission recorded
    assert admitted_under["billing_group_id"] == task_id
    assert admitted_under["billing_group_limit_source"] == "initial_task_admission"
    money = ("billing_group_id", "billing_group_limit_usd", "billing_group_limit_source",
             "billing_group_limit_revision")
    assert {key: continuation[key] for key in money} == {key: admitted_under.get(key) for key in money}
    after = load_task_result(tmp_path, task_id)
    assert after["continued_by"]["successor_task_id"] == row["id"] and after["continued_by"]["state"] == "admitted"
    assert after["unknown_unmetered"] is True and after["unresolved_upper_bound_usd"] == 1.5  # its money, unchanged


@pytest.mark.parametrize("terminal,refusal,named", [
    ({"cancel_origin": {"source": "http_single", "reason": "Stop now"}}, "stopped_by_owner", "http_single"),
    ({"cancel_origin": {"source": "panic"}}, "stopped_by_owner", "panic"),
    ({"deadline_at": "2000-01-01T00:00:00+00:00"}, "hard_limit_reached", "deadline"),
], ids=["stop", "panic", "deadline"])
@pytest.mark.parametrize("cause", CAUSES)
def test_a_refused_start_never_outranks_an_owner_stop_or_its_own_deadline(tmp_path, monkeypatch, cause, terminal,
                                                                         refusal, named):
    from ouroboros.owner_continue import continuation_offer
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation

    task_id, stored, workers, _frozen = _refused_root(tmp_path, monkeypatch, cause)
    seed_result(tmp_path, task_id, {**stored, **terminal})
    assert continuation_offer(load_task_result(tmp_path, task_id), task_id)["refusal"] == refusal
    outcome = admit_continuation(task_id, action_nonce=NONCE)
    assert outcome == {"ok": False, "error": refusal, "cause": named}
    assert workers.PENDING == [] and "continued_by" not in load_task_result(tmp_path, task_id)


@pytest.mark.parametrize("cause", CAUSES)
def test_a_refused_resume_whose_run_may_still_write_is_held_and_its_frozen_work_stays(tmp_path, monkeypatch, cause):
    """The started run of attempt 2 is still open: the Continue is accepted but held until that work settles, and
    the successor carries no saved cognition; the frozen source stays the predecessor's."""
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation
    from supervisor.events_budget import HOLD_CONTINUATION_WRITER

    task_id, _stored, workers, (locator, saved) = _refused_root(tmp_path, monkeypatch, cause, resumed=True)
    ack = admit_continuation(task_id, action_nonce=NONCE)
    assert ack["ok"] is True and ack["held"] is True, ack
    assert {(b["kind"], b.get("run_id")) for b in ack["blockers"]} >= {("delegated_run", "run-a1")}
    [row] = workers.PENDING
    assert row["id"] == ack["successor_task_id"] and row["task_contract"][KEY] is True
    assert "_working_recovery" not in row
    assert load_task_result(tmp_path, row["id"])["reason_code"] == HOLD_CONTINUATION_WRITER
    assert read_actor_source_bytes(tmp_path, task_id, locator["source_ref"]) == saved
