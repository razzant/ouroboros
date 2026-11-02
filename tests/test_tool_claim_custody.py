"""A returned call never stays «running»; an ended owner's local custody retires on positive evidence only.

The 28.10.2026 incident: a ``run_command`` returned in five seconds, its durable
claim could not be marked returned while the store was locked, the idle worker
that held it retired with no task (so nothing retired its claims), and macOS
gave its pid to another program, so Resume could never prove the owner dead.
Here: the process owes such a write until it lands (``custody_owed``); every
confirmed exit retires the exact process's claims across tasks and attempts;
startup and addressed actions retire only on positive evidence (kernel birth of
one kind, own pid of an earlier image, ESRCH, zombie); a known return never
regresses; a replica never carries custody. An owner or the mind may go on past
NAMED held calls whose outcome is unknown: recorded, never re-run, never beside a
live call (R8).
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from ouroboros import custody_owed, store, tool_custody
from ouroboros import local_custody_repair as repair
# Bind these modules' platform names BEFORE any test patches ``platform_layer``: a module first
# imported under a patch would keep the fake for every later test in this process.
from ouroboros import platform_layer, process_containment, process_custody, process_identity  # noqa: F401
from ouroboros.task_results import load_task_result, write_task_result

pytestmark = pytest.mark.serial

INCIDENT_BIRTH = "Tue Oct 27 14:57:17 2026"


def _claim(root, holder, op, *, pid, birth=INCIDENT_BIRTH, attempt=1, kernel="", state="claimed", extra=None,
           tree="root"):
    row = load_task_result(root, holder) or {}
    claims = dict(row.get("launch_handoffs") or {})
    claim = {"tool": "run_command", "task_id": holder, "root_task_id": tree, "state": state,
             "claimed_at": "2026-10-27T17:23:28+00:00",
             "local_owner": {"pid": pid, "process_birth": birth, "task_attempt": attempt}}
    if kernel:
        claim["owner_proof"] = {"pid": pid, "kernel_birth": kernel, "session": "an-earlier-session"}
    claim.update(extra or {})
    claims[op] = claim
    write_task_result(root, holder, row.get("status") or "completed", root_task_id=tree, launch_handoffs=claims)


def _claims(root, holder):
    return dict((load_task_result(root, holder) or {}).get("launch_handoffs") or {})


def _retired(root, holder):
    return dict((load_task_result(root, holder) or {}).get("retired_tool_invocations") or {})


def _busy():
    raise store.StoreBusy("store held by a long reader", reason="contention")


# --- owed facts ---------------------------------------------------------------------------------------------------


def test_an_owed_fact_stays_until_its_write_lands_and_a_newer_observation_replaces_it():
    attempts = []
    custody_owed.owe("row", lambda: attempts.append("old") or _busy(), label="op-1")
    assert custody_owed.drain() == 1 and custody_owed.labels() == ["op-1"], "a failed write keeps the duty"
    landed = []
    custody_owed.owe("row", lambda: landed.append(True), label="op-1")
    assert custody_owed.drain() == 0 and landed == [True] and custody_owed.labels() == []
    assert attempts == ["old"], "the replaced observation is never written again"


def test_one_failing_fact_never_starves_another():
    custody_owed.owe("a", lambda: (_ for _ in ()).throw(OSError("disk")), label="a")
    done = []
    custody_owed.owe("b", lambda: done.append(True), label="b")
    assert custody_owed.drain() == 1 and done == [True] and custody_owed.labels() == ["a"]


def test_a_process_that_owes_a_custody_write_is_not_quiescent():
    baseline = tool_custody.local_producers_settled()
    custody_owed.owe("row", _busy, label="op-1")
    assert tool_custody.local_producers_settled() is False, "idle retirement must not forget an owed write"
    custody_owed._FACTS.clear()
    assert tool_custody.local_producers_settled() is baseline


def test_the_pause_names_owed_writes_without_waiting_for_them(tmp_path):
    from ouroboros.budget_pause import local_producer_observation

    custody_owed.owe("row", _busy, label="op-owed")
    ctx = SimpleNamespace(task_id="paused-task", drive_root=str(tmp_path), budget_drive_root=str(tmp_path),
                          task_metadata={}, task_attempt=1)
    observed = local_producer_observation(ctx, timeout_sec=0)
    assert observed["publication_pending"] == ["op-owed"]
    custody_owed._FACTS.clear()
    assert local_producer_observation(ctx, timeout_sec=0)["publication_pending"] == []


# --- exit doors: the exact process, task or no task --------------------------------------------------------------


def test_an_exit_retires_every_claim_of_that_exact_process_across_tasks_and_attempts(tmp_path):
    write_task_result(tmp_path, "root", "completed", root_task_id="root")
    _claim(tmp_path, "t1", "op-a", pid=63888, attempt=1)
    _claim(tmp_path, "t2", "op-b", pid=63888, attempt=3)
    _claim(tmp_path, "t2", "op-other", pid=63888, birth="another process at the same pid")
    _claim(tmp_path, "t3", "op-receipt", pid=63888, state="returned",
           extra={"returned_at": "2026-10-27T17:33:12+00:00",
                  "control_receipts": [{"type": "steer_task", "routing_token": ""}]})
    tool_custody.retire_process_invocations(tmp_path, pid=63888, process_birth=INCIDENT_BIRTH)
    assert "op-a" not in _claims(tmp_path, "t1") and _retired(tmp_path, "t1")["op-a"]["effect_outcome"] == "unknown"
    assert _retired(tmp_path, "t1")["op-a"]["replay_authorized"] is False, "an unknown effect is never replayed"
    assert "op-b" not in _claims(tmp_path, "t2"), "attempt is a scope of the work, not of the process"
    assert "op-other" in _claims(tmp_path, "t2"), "another process at the same pid stays held"
    assert _claims(tmp_path, "t3")["op-receipt"]["state"] == "returned", "a known return never regresses"


def test_kernel_evidence_decides_when_both_sides_carry_it(tmp_path):
    _claim(tmp_path, "t1", "op-k", pid=7, kernel="darwin-start:1793102237.000001")
    tool_custody.retire_process_invocations(tmp_path, pid=7, process_birth=INCIDENT_BIRTH,
                                            kernel_birth="darwin-start:1793132901.017497")
    assert "op-k" in _claims(tmp_path, "t1"), "same pid and spelling but another kernel birth: not this process"
    tool_custody.retire_process_invocations(tmp_path, pid=7, process_birth="",
                                            kernel_birth="darwin-start:1793102237.000001")
    assert "op-k" not in _claims(tmp_path, "t1")


def test_a_confirmed_exit_is_owed_under_the_lock_and_written_by_the_drain(tmp_path, monkeypatch):
    from supervisor import worker_health

    _claim(tmp_path, "t1", "op-a", pid=63888)
    monkeypatch.setattr(worker_health, "_pool", lambda: SimpleNamespace(DRIVE_ROOT=tmp_path))
    alive = SimpleNamespace(proc=SimpleNamespace(pid=63888, exitcode=None, is_alive=lambda: True),
                            process_birth=INCIDENT_BIRTH, kernel_birth="")
    worker_health.owe_worker_exit(alive)
    assert custody_owed.pending() == 0, "a live process is never an exit"
    dead = SimpleNamespace(proc=SimpleNamespace(pid=63888, exitcode=0, is_alive=lambda: False),
                           process_birth=INCIDENT_BIRTH, kernel_birth="")
    worker_health.owe_worker_exit(dead)
    assert custody_owed.pending() == 1 and "op-a" in _claims(tmp_path, "t1"), "only memory under the queue lock"
    assert custody_owed.drain() == 0
    assert "op-a" not in _claims(tmp_path, "t1") and _retired(tmp_path, "t1")["op-a"]["state"] == "owner_dead"


def test_retirement_keeps_a_known_return_and_rewrites_nothing_twice(tmp_path):
    _claim(tmp_path, "t1", "op-r", pid=5, birth="b", state="returned",
           extra={"returned_at": "2026-10-27T17:33:12+00:00",
                  "control_receipts": [{"type": "steer_task", "routing_token": ""}]})
    _claim(tmp_path, "t1", "op-c", pid=5, birth="b")
    assert tool_custody.retire_process_invocations(tmp_path, pid=5, process_birth="b") == ["op-c"]
    assert _claims(tmp_path, "t1")["op-r"]["state"] == "returned" and "op-r" not in _retired(tmp_path, "t1")
    first = _retired(tmp_path, "t1")["op-c"]
    assert tool_custody.retire_process_invocations(tmp_path, pid=5, process_birth="b") == []
    assert _retired(tmp_path, "t1")["op-c"] == first, "an already retired row is not rewritten"


# --- positive evidence at startup and at addressed actions ------------------------------------------------------


def _platform(monkeypatch, *, gone=(), kernel=None, lstart="Tue Oct 27 23:28:21 2026"):
    monkeypatch.setattr(platform_layer, "pid_provably_gone", lambda pid: pid in set(gone))
    monkeypatch.setattr(process_containment, "pid_is_zombie", lambda _pid: False)
    monkeypatch.setattr(process_identity, "process_kernel_birth", lambda pid: (kernel or {}).get(pid))
    monkeypatch.setattr(platform_layer, "process_start_time", lambda _pid: lstart)


@pytest.mark.parametrize("live, recorded, ended", [
    ("darwin-start:1793132901.017497", "darwin-start:1793102237.000001", True),
    ("darwin-start:1793102237.000001", "darwin-start:1793102237.000001", False),
    ("123." + "a" * 32, "darwin-start:1793102237.000001", False),
    (None, "darwin-start:1793102237.000001", False),
])
def test_a_reused_pid_is_ended_only_by_kernel_evidence_of_one_kind(monkeypatch, live, recorded, ended):
    _platform(monkeypatch, kernel={63888: live})
    proof = {"pid": 63888, "kernel_birth": recorded, "session": "s"}
    assert repair.owner_ended(63888, INCIDENT_BIRTH, proof) is ended


def test_a_legacy_lstart_owner_on_a_reused_pid_stays_held(monkeypatch):
    _platform(monkeypatch, kernel={63888: "darwin-start:1793132901.017497"})
    assert repair.owner_ended(63888, INCIDENT_BIRTH) is False, "two ps renderings never prove another process"
    assert repair.owner_ended(63888, INCIDENT_BIRTH, {"pid": 63888, "kernel_birth": "", "session": "s"}) is False


def test_an_earlier_image_of_this_very_pid_ended_at_exec():
    from ouroboros.process_custody import current_custody_session_id

    proof = {"pid": os.getpid(), "kernel_birth": "", "session": "an-earlier-image"}
    assert repair.owner_ended(os.getpid(), "unvalidated", proof) is True
    assert repair.owner_ended(os.getpid(), "unvalidated", {**proof, "session": current_custody_session_id()}) is False


def test_startup_reconcile_retires_only_positively_ended_owners(tmp_path, monkeypatch):
    _platform(monkeypatch, gone={111}, kernel={222: "darwin-start:5.000005", 333: "darwin-start:9.000009"})
    write_task_result(tmp_path, "root", "completed", root_task_id="root")
    _claim(tmp_path, "t-dead", "op-dead", pid=111)
    _claim(tmp_path, "t-live", "op-live", pid=222, kernel="darwin-start:5.000005")
    _claim(tmp_path, "t-reused", "op-reused", pid=333, kernel="darwin-start:1.000001")
    _claim(tmp_path, "t-legacy", "op-legacy", pid=444)
    retired = repair.reconcile_open_local_custody(tmp_path)
    assert {row["task_id"] for row in retired} == {"t-dead", "t-reused"}
    assert "op-live" in _claims(tmp_path, "t-live"), "a live owner of one kernel birth stays held"
    assert "op-legacy" in _claims(tmp_path, "t-legacy"), "an unprovable legacy owner stays held"
    assert repair.reconcile_open_local_custody(tmp_path) == [], "the second pass rests"


def test_a_failed_retirement_is_owed_by_the_process_not_left_as_a_witness(tmp_path, monkeypatch):
    from ouroboros import obligations

    _claim(tmp_path, "t1", "op-a", pid=5, birth="b")
    retire = tool_custody.retire_process_invocations
    monkeypatch.setattr(tool_custody, "retire_process_invocations",
                        lambda *_a, **_k: (_ for _ in ()).throw(OSError("retirement disk failure")))
    assert repair.retire_ended_process(tmp_path, pid=5, process_birth="b")["owed"] is True
    assert custody_owed.labels() == ["exit:5"] and obligations.members(tmp_path, repair.WITNESS_SET) == {}
    monkeypatch.setattr(tool_custody, "retire_process_invocations", retire)
    assert custody_owed.drain() == 0
    assert _retired(tmp_path, "t1")["op-a"]["effect_outcome"] == "unknown"


# --- replicas carry results, not custody ------------------------------------------------------------------------


def test_a_replica_never_carries_custody_fields():
    from ouroboros.post_task_checkpoint import CANONICAL_CUSTODY_FIELDS, project_replica_task_result_fields

    canonical = {"task_id": "t", "status": "running", "launch_handoffs": {"op": {"state": "claimed"}}}
    replica = {"task_id": "t", "status": "running", "result": "child result", "launch_handoffs": {},
               "retired_tool_invocations": {"x": {"state": "owner_dead"}}, "retired_model_consumers": {"c": {}}}
    overlay = project_replica_task_result_fields(canonical, replica)
    assert not set(CANONICAL_CUSTODY_FIELDS) & set(overlay), "a stale replica can neither revive nor erase custody"
    assert overlay.get("result") == "child result", "results still flow"


# --- the owner reads what is known, never «has not returned» ------------------------------------------------------


def test_a_held_call_is_worded_by_what_is_actually_known(monkeypatch):
    from ouroboros.process_custody import current_custody_session_id
    from ouroboros.work_stop import resume_outcome

    _platform(monkeypatch, kernel={63888: "darwin-start:1793132901.017497"})
    own = {"pid": os.getpid(), "kernel_birth": "", "session": current_custody_session_id()}
    claims = {
        "live": {"tool": "run_command", "state": "claimed", "owner_proof": own,
                 "local_owner": {"pid": os.getpid(), "process_birth": "x", "task_attempt": 1}},
        "unknown": {"tool": "run_command", "state": "claimed",
                    "local_owner": {"pid": 63888, "process_birth": INCIDENT_BIRTH, "task_attempt": 1}},
        "ended": {"tool": "mcp_gitlab__get_commit", "state": "owner_dead",
                  "external_pending": {"t": "mcp_transport"}},
        "returned": {"tool": "tell_task", "state": "returned",
                     "control_receipts": [{"type": "steer_task", "routing_token": "r"}]},
    }
    reasons = {name: tool_custody.tool_handoff_reason(claim) for name, claim in claims.items()}
    assert reasons["live"]["reason"] == "local_call_live"
    assert reasons["unknown"]["reason"] == "local_completion_unknown", "a reused pid is not proof of life"
    assert reasons["ended"] == {"reason": "owner_ended_effect_unknown", "tool": "mcp_gitlab__get_commit",
                                "effects": ["MCP request"]}
    assert reasons["returned"]["reason"] == "returned_effect_unconfirmed"
    blockers = [{"kind": "tool_handoff", "task_id": "96ac2f7c981b4f54", **reason} for reason in reasons.values()]
    message = resume_outcome({"ok": False, "blockers": blockers})["message"]
    assert "has not returned" not in message
    assert "it is unknown whether the run_command call of task 96ac2f7c981b4f54 finished" in message
    assert "the outcome of its MCP request is unknown" in message and "its message to a task is not confirmed" in message


# --- going on past a named unknown outcome (R8) -----------------------------------------------------------------

LIVE_KERNEL = "darwin-start:5.000005"


def test_a_named_unknown_call_is_accepted_recorded_and_never_replayed(tmp_path, monkeypatch):
    _platform(monkeypatch, kernel={7: LIVE_KERNEL})
    _claim(tmp_path, "t1", "op-u", pid=63888)  # its worker can be proven neither live nor ended
    _claim(tmp_path, "t1", "op-live", pid=7, kernel=LIVE_KERNEL)
    _claim(tmp_path, "t1", "op-r", pid=5, state="returned", extra={"external_pending": {"tok": "mcp_transport"}})
    _claim(tmp_path, "t1", "op-other", pid=63888)
    by = {"actor": "owner", "text": "go on", "at": "2026-10-28T06:00:00+00:00"}
    outcome = repair.accept_unknown_custody(tmp_path, {"t1"}, ["op-u", "op-live", "op-r", "op-gone"],
                                            root_task_id="root", by=by)
    assert sorted(outcome["accepted"]) == ["op-r", "op-u"]
    assert outcome["refused"] == [{"id": "op-live", "task_id": "t1", "reason": "local_call_live"}], \
        "never a second writer beside a live call"
    assert outcome["unmatched"] == ["op-gone"]
    assert set(_claims(tmp_path, "t1")) == {"op-live", "op-other"}, "only the named rows move"
    fact = _retired(tmp_path, "t1")["op-u"]
    assert fact["state"] == "claimed", "kept as observed: a decision is not proof the worker ended"
    assert fact["effect_outcome"] == "unknown_accepted" and fact["replay_authorized"] is False
    assert fact["accepted"] == {**by, "reason": "local_completion_unknown"}
    assert _retired(tmp_path, "t1")["op-r"]["accepted"]["reason"] == "returned_effect_unconfirmed"
    from ouroboros.task_store import patch_handoff
    patch_handoff(tmp_path, "t1", "launch_handoffs", "op-u",
                  tool_custody._returned_claim(tmp_path, "2026-10-28T06:01:00+00:00", frozenset()))
    assert "op-u" not in _claims(tmp_path, "t1"), "a late return of an accepted call writes nothing"


def test_the_addressed_census_applies_a_bound_decision_to_its_own_tree_only(tmp_path, monkeypatch):
    from ouroboros.control_actor import accepting_unknown, mind_control
    from supervisor import continuation_admission as admission

    _platform(monkeypatch)
    for tree in ("A", "B"):
        _claim(tmp_path, tree, f"op-{tree}", pid=63888, tree=tree)
    monkeypatch.setattr(admission, "conflicting_writers", lambda *_a, **_k: [])
    events = []
    q = SimpleNamespace(DRIVE_ROOT=tmp_path, append_jsonl=lambda _path, row: events.append(row))
    admission.action_writers(q, "A", drive_root=tmp_path)
    assert "op-A" in _claims(tmp_path, "A") and not events, "no decision, nothing accepted"
    with mind_control("mind-task", "the owner asked me to finish"), accepting_unknown(["op-A", "op-B"],
                                                                                     "the owner asked me to finish"):
        admission.action_writers(q, "A", drive_root=tmp_path)
    assert "op-A" not in _claims(tmp_path, "A") and "op-B" in _claims(tmp_path, "B"), "only the addressed tree"
    [event] = [row for row in events if row["type"] == "local_custody_unknown_accepted"]
    assert event["accepted"] == ["op-A"] and event["unmatched"] == ["op-B"]
    assert _retired(tmp_path, "A")["op-A"]["accepted"]["actor"] == "ouroboros"
    assert _retired(tmp_path, "A")["op-A"]["accepted"]["task_id"] == "mind-task"


def test_a_receiver_of_unknown_fate_is_accepted_and_leaves_the_census(tmp_path, monkeypatch):
    from ouroboros import model_wait, usage_accounting as ua
    from supervisor.continuation_admission import model_handoffs

    write_task_result(tmp_path, "root", "running", root_task_id="root",
                      billing_group={"billing_group_id": "root", "billing_group_limit_usd": 20.0,
                                     "billing_group_limit_source": "initial_task_admission",
                                     "billing_group_limit_revision": "admission-1"})
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    owner = model_wait.TaskModelWait(task={"id": "root", "_attempt": 2}, drive_root=tmp_path,
                                     event_queue=None, worker_slot_held=True)
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="root", root_task_id="root")):
        with model_wait.operation_wait_scope(owner):
            def send():
                raise RuntimeError("response unknown")
            with pytest.raises(RuntimeError):
                ua.execute_physical_attempt(ua.AttemptRequest(model="m", provider="test", reservation_usd=.2), send)
    [held] = model_handoffs(tmp_path, "root")
    assert held["reason"] == "local_call_live", "this very process still receives it"
    monkeypatch.setattr(repair, "owner_live", lambda *_a, **_k: False)
    [held] = model_handoffs(tmp_path, "root")
    assert held["reason"] == "local_completion_unknown"
    by = {"actor": "owner", "text": "", "at": "2026-10-28T06:00:00+00:00"}
    outcome = repair.accept_unknown_custody(tmp_path, {"root"}, [held["attempt_id"]], root_task_id="root", by=by)
    assert outcome["accepted"] == [held["attempt_id"]]
    assert model_handoffs(tmp_path, "root") == [], "the receiver is retired; its money stays where it was"
    retired = load_task_result(tmp_path, "root")["retired_model_consumers"][owner.answer_consumer_id]
    assert retired["accepted"]["reason"] == "local_completion_unknown" and retired["task_attempt"] == 2
    assert repair.accept_unknown_custody(tmp_path, {"root"}, [held["attempt_id"]], root_task_id="root",
                                         by=by)["accepted"] == [], "nothing is recorded twice"


def test_only_a_census_of_unknown_outcomes_offers_going_on_and_every_door_carries_the_ids(monkeypatch):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.control_actor import accepted_unknown
    from ouroboros.gateway.tasks import api_task_resume
    from ouroboros.work_stop import acceptable_unknown, resume_outcome
    from supervisor import events_mind_controls

    unknown = {"kind": "tool_handoff", "task_id": "t", "operation_id": "op-u", "reason": "local_completion_unknown"}
    receiver = {"kind": "model_handoff", "task_id": "t", "attempt_id": "at-u", "reason": "local_completion_unknown"}
    live = {**unknown, "operation_id": "op-l", "reason": "local_call_live"}
    assert acceptable_unknown([unknown, receiver]) == ["op-u", "at-u"]
    assert acceptable_unknown([unknown, live]) == [] and acceptable_unknown([]) == []
    assert acceptable_unknown([unknown, {"kind": "running_member", "task_id": "c"}]) == []
    refusal = resume_outcome({"ok": False, "error": "owner_pause_effects_unsettled", "blockers": [unknown, receiver]})
    assert refusal["acceptable_unknown"] == ["op-u", "at-u"]
    assert "unknown whether the model answer receiver of task t ended" in refusal["message"]

    seen = []
    monkeypatch.setattr("supervisor.queue.resume_budget_paused_task",
                        lambda _id: seen.append(accepted_unknown()) or {**refusal, "error": refusal["error"]})
    with TestClient(Starlette(routes=[Route("/api/tasks/{task_id}/resume", api_task_resume, methods=["POST"])])) as client:
        plain = client.post("/api/tasks/t/resume")
        named = client.post("/api/tasks/t/resume", json={"accept_unknown": ["op-u"], "reason": "go on"})
        wrong = client.post("/api/tasks/t/resume", json={"accept_unknown": "op-u"})
    assert plain.status_code == named.status_code == 409 and plain.json()["acceptable_unknown"] == ["op-u", "at-u"]
    assert seen == [{}, {"ids": ["op-u"], "text": "go on"}], "the owner's decision is bound to that one call"
    assert wrong.status_code == 400

    def observed(*_a, **_k):
        seen.append(accepted_unknown())
        return {"ok": True}
    monkeypatch.setattr("supervisor.queue_transitions.resume_budget_paused_task", observed)
    monkeypatch.setattr("supervisor.continuation_admission.admit_continuation", observed)
    monkeypatch.setattr("ouroboros.work_diagnostics.control_actor_reason", lambda *_a, **_k: "")
    seen.clear()
    for action in ("resume_task", "continue_task"):
        events_mind_controls._apply({"task_id": "t", "actor_task_id": "mind", "action_nonce": "n", "action": action,
                                     "reason": "the owner asked", "accept_unknown": ["op-u"]})
    assert seen == [{"ids": ["op-u"], "text": "the owner asked"}, {}], "only resume_task carries the decision"


# --- review round 1: one writer by process, drivers and doors exercised for real ---------------------------------


def test_an_earlier_image_of_this_pid_is_retired_and_this_images_claims_never_are(tmp_path, monkeypatch):
    """An exec keeps pid, births and attempt; only the custody session tells the images apart."""
    from ouroboros import model_wait, usage_accounting as ua

    here = tool_custody.owner_proof()
    write_task_result(tmp_path, "r1", "running", root_task_id="r1",
                      billing_group={"billing_group_id": "r1", "billing_group_limit_usd": 20.0,
                                     "billing_group_limit_source": "initial_task_admission",
                                     "billing_group_limit_revision": "admission-1"})
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    receiver = model_wait.TaskModelWait(task={"id": "r1", "_attempt": 1}, drive_root=tmp_path,
                                        event_queue=None, worker_slot_held=True)  # this image's live receiver
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="r1", root_task_id="r1")):
        with model_wait.operation_wait_scope(receiver):
            with pytest.raises(RuntimeError, match="response unknown"):
                ua.execute_physical_attempt(ua.AttemptRequest(model="m", provider="t", reservation_usd=.2),
                                            lambda: (_ for _ in ()).throw(RuntimeError("response unknown")))
    [bound] = [row for row in ua.read_usage_records(tmp_path, final_only=True)
               if row.get("local_answer_consumer_id") == receiver.answer_consumer_id]
    assert bound["local_answer_owner_pid"] == os.getpid(), "the receiver is really bound by this image"
    kernel = here["kernel_birth"] or "darwin-start:1.000001"
    _claim(tmp_path, "t1", "op-old", pid=os.getpid(), birth="b", kernel=kernel)  # session: an earlier image's
    _claim(tmp_path, "t1", "op-now", pid=os.getpid(), birth="b", extra={"owner_proof": {**here, "kernel_birth": kernel}})
    _claim(tmp_path, "t2", "op-legacy", pid=os.getpid(), birth="b")  # an older release wrote no session at all
    retired = repair.reconcile_open_local_custody(tmp_path)
    assert {row["task_id"] for row in retired} == {"t1"}, "one proof retires that process's claims in every task"
    assert "op-old" not in _claims(tmp_path, "t1") and "op-legacy" not in _claims(tmp_path, "t2")
    assert "op-now" in _claims(tmp_path, "t1"), "a claim this live image wrote is never retired"
    retired_receivers = load_task_result(tmp_path, "r1").get("retired_model_consumers") or {}
    assert receiver.answer_consumer_id not in retired_receivers, "nor a receiver this live image bound"
    assert tool_custody.retire_process_invocations(tmp_path, pid=os.getpid(), process_birth="b",
                                                   kernel_birth=kernel) == [], "not even by an exit-shaped selector"
    assert repair.owner_live(os.getpid(), "b", here) and not repair.owner_live(os.getpid(), "b", {"pid": os.getpid()})


def test_a_confirmation_owed_after_its_claim_settled_away_is_discharged(tmp_path):
    write_task_result(tmp_path, "t1", "running", root_task_id="root")
    remove = tool_custody._without_token("tok")
    custody_owed.owe(("confirm", "gone-op"), lambda: tool_custody._patch_owed(tmp_path, "t1", "gone-op", remove),
                     label="gone-op")
    assert custody_owed.drain() == 0, "an absent claim is done, never retried forever"
    assert "gone-op" not in _claims(tmp_path, "t1"), "and never recreated"
    custody_owed.owe(("confirm", "no-task"), lambda: tool_custody._patch_owed(tmp_path, "no-task", "op", remove),
                     label="op")
    assert custody_owed.drain() == 0, "a task whose row is gone has nothing left to change"


def test_rebinding_the_same_wake_does_not_rerun_its_duty():
    import threading

    custody_owed.owe("a fact a busy store refuses", _busy, label="busy")
    wake = threading.Event()
    custody_owed.bind_wake(wake.set)
    assert wake.is_set(), "a fact owed before the first bind wakes the duty once"
    wake.clear()
    custody_owed.bind_wake(wake.set)  # the server re-binds on every tick
    assert not wake.is_set(), "a fact that keeps failing waits for its driver's own cadence"
    custody_owed.owe("another fact", lambda: None, label="next")
    assert not wake.is_set(), "one wake per undrained batch: the driver has not drained since"
    assert custody_owed.drain() == 1, "the busy fact stays; the drain re-arms the wake"
    custody_owed.owe("a later fact", lambda: None, label="later")
    assert wake.is_set(), "what is owed after a drain wakes the driver again"


def test_the_servers_delivery_duty_writes_what_the_process_owes(tmp_path, monkeypatch):
    import threading

    from ouroboros import server_maintenance

    monkeypatch.setattr("ouroboros.dialogue_retry.drain", lambda: None)
    monkeypatch.setattr("supervisor.terminal_delivery.replay_pending_deliveries", lambda _root: None)
    _claim(tmp_path, "t1", "op-a", pid=63888)
    repair.retire_exited_process(tmp_path, pid=63888, process_birth=INCIDENT_BIRTH)
    latch = threading.Lock()
    latch.acquire()
    server_maintenance._run_terminal_duty(tmp_path, "terminal_delivery_replay", latch)
    assert custody_owed.pending() == 0 and "op-a" not in _claims(tmp_path, "t1")


def test_the_startup_sweep_retires_an_ended_owners_claims_after_the_reaper(tmp_path, monkeypatch):
    from ouroboros import server_maintenance

    order = []
    monkeypatch.setattr(server_maintenance, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server_maintenance, "_installed_skill_names", lambda: set())
    monkeypatch.setattr("ouroboros.process_custody.reap_orphaned_processes",
                        lambda *_a, **_k: order.append("reaper") or [])
    monkeypatch.setattr("ouroboros.process_scope.settle_task_scopes", lambda *_a, **_k: None)
    monkeypatch.setattr(server_maintenance, "_reconcile_delegated_runs", lambda *_a, **_k: None)
    monkeypatch.setattr("ouroboros.delegate_terminal.backfill_terminal_reconciliations", lambda _root: [])
    monkeypatch.setattr("supervisor.terminal_delivery.replay_pending_deliveries", lambda _root: None)
    reconcile = repair.reconcile_open_local_custody
    monkeypatch.setattr(repair, "reconcile_open_local_custody", lambda root: order.append("custody") or reconcile(root))
    _platform(monkeypatch, gone={111})
    _claim(tmp_path, "t1", "op-dead", pid=111)
    _claim(tmp_path, "t1", "op-unknown", pid=63888)
    server_maintenance._startup_custody_sweep()
    assert order == ["reaper", "custody"]
    assert "op-dead" not in _claims(tmp_path, "t1") and "op-unknown" in _claims(tmp_path, "t1")


def test_replacing_the_pools_last_dead_member_owes_its_exit(tmp_path, monkeypatch):
    from supervisor import worker_pool_lifecycle as lifecycle, workers

    from tests.test_owner_wait_pool import InputQueue, Process

    dead = SimpleNamespace(wid=0, proc=Process(42), in_q=InputQueue(), process_birth=INCIDENT_BIRTH, kernel_birth="")
    dead.proc.alive, dead.proc.exitcode = False, 0
    monkeypatch.setattr(workers, "WORKERS", {0: dead})
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr(workers, "_get_ctx", lambda: SimpleNamespace(Queue=InputQueue, Process=lambda **_kw: Process(43)))
    monkeypatch.setattr(workers, "get_event_q", InputQueue)
    monkeypatch.setattr("supervisor.worker_process.spawn_worker_process", lambda *_a, **_k: Process(43))
    monkeypatch.setattr("supervisor.worker_process.close_worker_stop_channel", lambda _proc: None)
    monkeypatch.setattr(lifecycle.threading, "Thread", lambda **_kw: SimpleNamespace(start=lambda: None))
    _claim(tmp_path, "t1", "op-a", pid=42)
    assert lifecycle.retire_or_replace_worker(0, dead) == "replaced"
    assert custody_owed.labels() == ["exit:42"], "a replacement door owes the ended member's exit too"
    assert custody_owed.drain() == 0 and "op-a" not in _claims(tmp_path, "t1")


# --- review round 2 ------------------------------------------------------------------------------------------------


def test_a_request_stamp_of_this_live_process_is_never_ended_by_the_custody_session_rule():
    """``workspace_preparation`` judges request owners with no ``owner_proof``: the birth comparison stays."""
    own = platform_layer.process_start_time(os.getpid())
    assert repair.owner_ended(os.getpid(), own) is False, "a stamp without a session is not a custody row"
    assert repair.owner_ended(os.getpid(), own, {}) is True, "a custody row without this image's session is"
    assert repair.owner_ended(os.getpid(), own, tool_custody.owner_proof()) is False


def test_a_worker_parked_for_its_owner_services_what_it_owes(monkeypatch):
    import queue as stdqueue

    from ouroboros import owner_wait
    from ouroboros.model_wait import ModelWaitInterrupted

    written, polls = [], []

    def write():
        written.append(1)
        if len(written) == 1:
            raise OSError("the store is busy")

    def hold(_ctx):
        polls.append(1)
        return "test_stop" if written == [1, 1] or len(polls) > 50 else ""  # fails, never hangs, without the drain
    custody_owed.owe("late", write, label="late-call")
    monkeypatch.setattr("ouroboros.budget_pause._hold_control_reason", hold)
    ctx = SimpleNamespace(task_id="t1", task_attempt=1, pending_events=[], drive_root="/nonexistent")
    empty = SimpleNamespace(get=lambda timeout: (_ for _ in ()).throw(stdqueue.Empty()))
    with pytest.raises(ModelWaitInterrupted):
        owner_wait.worker_owner_wait(0, empty, SimpleNamespace(put=lambda _m: None), ctx, {"wait_id": "w"})
    assert written == [1, 1] and custody_owed.pending() == 0, "the park's poll retried and landed it"


def test_the_named_decision_applies_even_when_the_repair_before_it_fails(tmp_path, monkeypatch):
    from ouroboros.control_actor import accepting_unknown
    from supervisor import continuation_admission as admission

    _platform(monkeypatch)
    _claim(tmp_path, "A", "op-A", pid=63888, tree="A")
    monkeypatch.setattr(admission, "conflicting_writers", lambda *_a, **_k: [])
    monkeypatch.setattr(repair, "repair_ended_local_custody",
                        lambda *_a, **_k: (_ for _ in ()).throw(OSError("repair refused")))
    q = SimpleNamespace(DRIVE_ROOT=tmp_path, append_jsonl=lambda _path, _row: None)
    with accepting_unknown(["op-A"], "go on"):
        admission.action_writers(q, "A", drive_root=tmp_path)
    assert _retired(tmp_path, "A")["op-A"]["effect_outcome"] == "unknown_accepted"


def test_one_refusing_row_never_starves_the_other_tasks_of_an_ended_process(tmp_path, monkeypatch):
    _claim(tmp_path, "t-bad", "op-bad", pid=63888)
    _claim(tmp_path, "t-good", "op-good", pid=63888)
    retire = tool_custody._retire_owned_claims

    def refusing(root, holder, owned):
        if holder == "t-bad":
            raise ValueError("tool_invocation_authority_unreadable")
        return retire(root, holder, owned)
    monkeypatch.setattr(tool_custody, "_retire_owned_claims", refusing)
    assert repair.retire_ended_process(tmp_path, pid=63888, process_birth=INCIDENT_BIRTH)["owed"] is True
    assert "op-good" not in _claims(tmp_path, "t-good"), "the other task's claim retired anyway"
    assert "op-bad" in _claims(tmp_path, "t-bad") and custody_owed.labels() == ["exit:63888"]


def test_a_receiver_retirement_owed_for_a_task_that_is_gone_is_discharged(tmp_path):
    from ouroboros import model_wait

    model_wait._owe_consumer_retirement(tmp_path, "gone-task", {"consumer": 1})
    assert custody_owed.drain() == 0, "nothing is left to retire on a task whose row is gone"


def test_two_unconfirmed_effects_are_worded_in_the_plural():
    from ouroboros.work_stop import tool_handoff_words

    words = tool_handoff_words({"task_id": "t", "tool": "run_command", "reason": "returned_effect_unconfirmed",
                                "effects": ["started process", "MCP request"]})
    assert words.endswith("its started process and MCP request are not confirmed")


def test_retiring_a_non_last_dead_member_owes_its_exit(tmp_path, monkeypatch):
    """The incident's own door: idle retirement of a member of a pool that keeps others."""
    from supervisor import worker_pool_lifecycle as lifecycle, workers

    from tests.test_owner_wait_pool import InputQueue, Process

    dead = SimpleNamespace(wid=0, proc=Process(42), in_q=InputQueue(), process_birth=INCIDENT_BIRTH, kernel_birth="")
    dead.proc.alive, dead.proc.exitcode = False, 0
    other = SimpleNamespace(wid=1, proc=Process(43), in_q=InputQueue())
    monkeypatch.setattr(workers, "WORKERS", {0: dead, 1: other})
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr("supervisor.worker_process.close_worker_stop_channel", lambda _proc: None)
    _claim(tmp_path, "t1", "op-a", pid=42)
    assert lifecycle.retire_or_replace_worker(0, dead) == "retired"
    assert custody_owed.labels() == ["exit:42"]
    assert custody_owed.drain() == 0 and "op-a" not in _claims(tmp_path, "t1")



def test_a_decision_that_lands_on_one_row_is_journalled_even_when_another_row_refuses(tmp_path, monkeypatch):
    from ouroboros import task_store
    from ouroboros.control_actor import accepting_unknown
    from supervisor import continuation_admission as admission

    _platform(monkeypatch)
    _claim(tmp_path, "A", "op-A", pid=63888, tree="A")
    _claim(tmp_path, "A-child", "op-child", pid=63888, tree="A")
    patch = task_store.patch_document
    monkeypatch.setattr(task_store, "patch_document", lambda root, holder, mutator: (
        (_ for _ in ()).throw(OSError("busy")) if holder == "A-child" else patch(root, holder, mutator)))
    monkeypatch.setattr(admission, "conflicting_writers", lambda *_a, **_k: [])
    events = []
    q = SimpleNamespace(DRIVE_ROOT=tmp_path, append_jsonl=lambda _path, row: events.append(row))
    with accepting_unknown(["op-A", "op-child"], "go on"):
        admission.action_writers(q, "A", drive_root=tmp_path)
    [event] = [row for row in events if row["type"] == "local_custody_unknown_accepted"]
    assert event["accepted"] == ["op-A"] and event["refused"] == [
        {"id": "op-child", "task_id": "A-child", "reason": "unwritten"}]
    assert "op-child" in _claims(tmp_path, "A-child"), "the refused row stays held as it was"


def test_a_retirement_never_overwrites_a_recorded_decision(tmp_path):
    from ouroboros import model_wait

    write_task_result(tmp_path, "t1", "running", root_task_id="root")
    by = {"actor": "owner", "text": "go on", "at": "2026-10-28T07:00:00+00:00"}
    model_wait.retire_model_consumers(tmp_path, "t1", {"c": 1}, record={"accepted": by})
    model_wait.retire_model_consumers(tmp_path, "t1", {"c": 1})  # a death proof that read before the decision
    assert load_task_result(tmp_path, "t1")["retired_model_consumers"]["c"]["accepted"] == by


def test_a_busy_store_ends_the_pass_at_once_instead_of_trying_every_row(tmp_path, monkeypatch):
    for holder in ("t1", "t2", "t3"):
        _claim(tmp_path, holder, f"op-{holder}", pid=63888)
    attempts = []

    def busy(root, holder, owned):
        attempts.append(holder)
        _busy()
    monkeypatch.setattr(tool_custody, "_retire_owned_claims", busy)
    with pytest.raises(store.StoreBusy):
        tool_custody.retire_process_invocations(tmp_path, pid=63888, process_birth=INCIDENT_BIRTH)
    assert attempts == ["t1"], "contention is store-wide: the next row would meet the same lock"
    _platform(monkeypatch, gone={63888})
    attempts.clear()
    repair.reconcile_open_local_custody(tmp_path)
    assert attempts == ["t1"], "one pass tries an ended process once, not once per holder"


def test_a_landed_decision_is_reported_even_when_the_receivers_cannot_be_read(tmp_path, monkeypatch):
    _platform(monkeypatch)
    _claim(tmp_path, "t1", "op-u", pid=63888)
    monkeypatch.setattr("ouroboros.owner_pause.tree_model_requests",
                        lambda *_a, **_k: (_ for _ in ()).throw(OSError("usage store unreadable")))
    outcome = repair.accept_unknown_custody(tmp_path, {"t1"}, ["op-u", "attempt-x"], root_task_id="root",
                                            by={"actor": "owner", "text": "", "at": "2026-10-28T07:30:00+00:00"})
    assert outcome["accepted"] == ["op-u"], "what landed is reported, so its journal event is written"
    assert outcome["refused"] == [{"id": "attempt-x", "task_id": "", "reason": "unreadable"}]


def test_an_undelivered_wake_is_tried_again_by_the_next_owe():
    calls = []

    def wake():
        calls.append(1)
        if len(calls) == 1:
            raise OSError("the queue is closing")
    custody_owed.bind_wake(wake)
    custody_owed.owe("first", _busy, label="first")
    custody_owed.owe("second", _busy, label="second")
    assert calls == [1, 1], "a wake that failed to deliver is not counted as sent"
