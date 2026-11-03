"""A Presence attempt is observed only while something can still write it.

The work view is ``pending`` while the attempt's producer can still change it and terminal
once it cannot; "outcome unknown" travels as facts on that terminal view, never as a state.
The orphan reconciler settles a dead executor's RUNNING row from the same rule.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from ouroboros import presence_runner
from ouroboros.platform_layer import process_start_time
from ouroboros.presence_recovery import attempt_ended, record_unresolved_turn, retained_work_view
from ouroboros.presence_runner import PresenceTurnError, PresenceTurnGate, _build_task, _cached_result, run_presence_turn
from ouroboros.process_scope import task_scope_terminal_evidence
from ouroboros.review_operation import controller_identity
from ouroboros.task_results import is_reconciled_presence_placeholder, load_task_result, write_task_result
from ouroboros.task_status import (effective_task_result, execution_owner_record, reconcile_orphaned_running_tasks,
                                    wait_for_effective_tasks)
from ouroboros.task_store import patch_document
from ouroboros.terminal_projection import terminal_projection_owed
from tests import _dialogue as dialogue_test
from tests.test_presence_runner import _admission, _event

OUTPUT = {"output_ref": "presence-output-early", "outcome": "message", "text": "Here is the report."}


def _dead_witness() -> dict:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return {"pid": child.pid, "birth": "gone", "session": "previous-generation"}


def _alive_witness() -> dict:
    parent = os.getppid()
    return {"pid": parent, "birth": process_start_time(parent), "session": "another-process"}


def _parked(witness: dict) -> dict:
    return {"version": 1, "initial": {"status": "continuing", "outcome": "message", "text": OUTPUT["text"],
                                      "output_ref": OUTPUT["output_ref"], "work_ref": ""},
            "outputs": [OUTPUT], "child_work_ref": "child-1", "author_controller": witness}


def _row(root, event_id, status="running", *, owner=None, parked=None, recorded=False, unknown=False,
         answer="", child=""):
    task = _build_task(_admission(), replace(_event(), source_event_id=event_id, text="Is the report ready?"),
                       drive_root=root, staged_files=())
    metadata = copy.deepcopy(task["metadata"])
    metadata.update(presence_outcome="message" if answer else "silent", presence_result_text=answer)
    if unknown:
        metadata["presence_unknown_outcome"] = {"source": "provider_outcome_unknown_no_resend"}
    if child:
        metadata["presence_work_ref"] = child
    write_task_result(root, task["id"], status, metadata=metadata, chat_id=task["chat_id"], _is_direct_chat=True,
                      source="presence", result=answer or "Task is running.",
                      terminal_origin="model_final" if status in {"completed", "failed", "cancelled"} else "",
                      **({"execution_owner": owner(task)} if owner else {}))
    if parked is not None:
        patch_document(root, task["id"], lambda current: {**current, "presence_continuation": parked})
    if recorded:
        assert record_unresolved_turn(root, task["id"])  # the runner's gate proof and its one room fact
    return task["id"], load_task_result(root, task["id"])


def _room_facts(root, ref):
    return [row for row in dialogue_test.log_rows(root / "logs" / "chat.jsonl")
            if row.get("summary_id") == f"presence-retained:{ref}"]


def test_new_presence_rows_name_their_controller():
    owner = execution_owner_record("/tmp", {"id": "presence-x"}, "presence")
    assert owner["kind"] == "presence" and owner["controller"] == controller_identity()
    assert "controller" not in execution_owner_record("/tmp", {"id": "pooled-x"}, "pooled")


def test_cancelled_turn_ends_with_its_facts_released_outputs_and_child(tmp_path):
    ref, stored = _row(tmp_path, "ev-cancelled", "cancelled", parked=_parked(_dead_witness()), child="child-1")
    status, view = retained_work_view(stored, ref, tmp_path)
    assert status == 200 and view["status"] == "cancelled"
    assert (view["outcome"], view["text"], view["output_ref"]) == ("silent", "", "")
    # The polled reference names itself; the real child rides beside it, as every transport reads it.
    assert view["work_ref"] == view["turn_ref"] == ref and view["child_work_ref"] == "child-1"
    assert view["outputs"] == [OUTPUT] and view["continuation_version"] == 1
    assert view["code"] == "presence_turn_cancelled" and view["recorded_status"] == "cancelled"
    assert "disposition" not in view


def _witnessed(root, witness):
    return lambda task: {**execution_owner_record(root, task, "presence"), "controller": witness}


@pytest.mark.parametrize("cause", ["gate_proof", "dead_controller", "dead_parked_author",
                                   "local_runner_saw_it_end", "returned_in_live_worker"])
def test_ended_attempt_is_terminal_then_settled_once(tmp_path, cause):
    # The runner's own end marker is an attempt fact: a still-living worker process cannot hold it open.
    witness = {"dead_controller": _dead_witness(), "local_runner_saw_it_end": controller_identity(),
               "returned_in_live_worker": _alive_witness()}.get(cause)
    ref, stored = _row(tmp_path, f"ev-{cause}",
                       owner=_witnessed(tmp_path, witness) if witness else None,
                       parked=_parked(_dead_witness()) if cause == "dead_parked_author" else None,
                       recorded=cause in {"gate_proof", "local_runner_saw_it_end", "returned_in_live_worker"})
    assert attempt_ended(stored, tmp_path)
    status, view = retained_work_view(stored, ref, tmp_path)
    assert status == 200 and view["status"] == "failed" and view["work_ref"] == ref
    assert view["code"] == "presence_attempt_outcome_unknown" and view["recorded_status"] == "running"
    assert view["text"] == "" and view["outcome"] == "silent"
    if cause == "dead_parked_author":
        assert view["outputs"] == [OUTPUT] and view["child_work_ref"] == "child-1"
    effective = effective_task_result(tmp_path, stored, materialize_artifacts=False)
    assert effective["status"] == "failed" and effective["status_reconciled_from"] == "running"
    assert reconcile_orphaned_running_tasks(tmp_path) == 1
    settled = load_task_result(tmp_path, ref)
    assert (settled["status"], settled["reason_code"], settled["status_reconciled_from"]) == (
        "failed", "presence_attempt_lost", "running")
    # An ordinary terminal, not a reopenable placeholder: it owes its room projection and ends its processes.
    assert not is_reconciled_presence_placeholder(settled)
    assert terminal_projection_owed(ref, settled) and task_scope_terminal_evidence(settled)
    assert reconcile_orphaned_running_tasks(tmp_path) == 0
    assert len(_room_facts(tmp_path, ref)) == 1  # the mind learns once, whoever proved the end
    after = retained_work_view(settled, ref, tmp_path)[1]
    assert after["status"] == "failed" and after["status_reconciled_from"] == "running"
    with pytest.raises(PresenceTurnError) as refused:  # a resubmission never regenerates
        _cached_result(tmp_path, ref)
    assert (refused.value.disposition, refused.value.code) == ("retained", "presence_attempt_outcome_unknown")


def test_a_recorded_interrupted_alone_proves_no_end(tmp_path):
    ref, stored = _row(tmp_path, "ev-interrupted", "interrupted")
    assert not attempt_ended(stored, tmp_path)
    assert retained_work_view(stored, ref, tmp_path)[1]["status"] == "pending"
    assert reconcile_orphaned_running_tasks(tmp_path) == 0


@pytest.mark.parametrize("status,unknown", [("cancelled", False), ("failed", True)])
@pytest.mark.parametrize("parked", [False, True])
def test_live_execution_outranks_a_recorded_unanswered_terminal(tmp_path, status, unknown, parked):
    # A Stop written under a live author (d375d9fd0's miss lane, V5), or an unknown outcome stored before
    # handle_task returns: the execution may still back-fill a child, so it stays observed until it ends.
    ref, stored = _row(tmp_path, f"ev-live-{status}-{parked}", status, unknown=unknown, child="child-1",
                       parked=_parked(controller_identity()) if parked else None)
    key = (str(tmp_path.resolve()), ref)
    presence_runner._LIVE_PRESENCE_TASKS.add(key)
    try:
        assert not attempt_ended(stored, tmp_path)
        code, pending = retained_work_view(stored, ref, tmp_path)
        assert code == 202 and pending["status"] == "pending"
        assert not {"outcome", "text", "output_ref"} & pending.keys()
        if parked:
            assert pending["outputs"] == [OUTPUT] and pending["work_ref"] == ref
    finally:
        presence_runner._LIVE_PRESENCE_TASKS.discard(key)
    code, view = retained_work_view(stored, ref, tmp_path)
    assert code == 200 and view["status"] == status and view["child_work_ref"] == "child-1"
    assert view["code"] and (view["outcome"], view["text"]) == ("silent", "")


def test_a_completed_turn_with_a_workspace_stays_completed(tmp_path):
    # Inline Presence has no workspace artifact finalization: its recorded terminal is final on every read,
    # even after its controller died (it must not read as running, nor be re-projected as lost).
    ref, stored = _row(tmp_path, "ev-workspace-answered", "completed", answer="Done.",
                       owner=_witnessed(tmp_path, _dead_witness()))
    patch_document(tmp_path, ref, lambda current: {**current, "workspace_root": str(tmp_path / "external")})
    stored = load_task_result(tmp_path, ref)
    effective = effective_task_result(tmp_path, stored, materialize_artifacts=False)
    assert effective["status"] == "completed" and "status_reconciled_from" not in effective
    assert reconcile_orphaned_running_tasks(tmp_path) == 0
    assert retained_work_view(stored, ref, tmp_path)[1]["text"] == "Done."
    # A root promoted from the turn speaks from the same carrier but is pooled work: its workspace
    # artifacts still finalize before it reads terminal, and a wait keeps waiting until they do.
    write_task_result(tmp_path, "promoted-root", "completed", metadata={"presence": stored["metadata"]["presence"]},
                      source="presence_promote", delegation_role="root", root_task_id="promoted-root",
                      execution_owner=execution_owner_record(tmp_path, {"id": "promoted-root"}, "pooled"),
                      workspace_root=str(tmp_path / "external"), artifact_status="finalizing", result="Worker done.")
    promoted = effective_task_result(tmp_path, load_task_result(tmp_path, "promoted-root"), materialize_artifacts=False)
    assert (promoted["status"], promoted["child_status"], promoted["artifact_status"]) == (
        "running", "completed", "finalizing")
    assert wait_for_effective_tasks(tmp_path, ["promoted-root"], timeout_sec=0)["all_terminal"] is False


def test_an_executor_that_raises_records_its_end_and_is_settled_once(tmp_path):
    event = replace(_event(), source_event_id="ev-raised", text="Is the report ready?")

    class Agent:
        def handle_task(self, task):
            raise RuntimeError("context assembly failed after the start barrier")

    with pytest.raises(RuntimeError):
        run_presence_turn(admission=_admission(), event=event, repo_dir=tmp_path, drive_root=tmp_path,
                          agent_factory=lambda **_kwargs: Agent(), gate=PresenceTurnGate(1))
    ref = presence_runner._task_id(_admission(), event)
    stored = load_task_result(tmp_path, ref)
    assert stored["status"] == "running" and stored["presence_recovery_recorded"]
    assert len(_room_facts(tmp_path, ref)) == 1
    assert attempt_ended(stored, tmp_path)
    assert retained_work_view(stored, ref, tmp_path)[1]["status"] == "failed"
    assert reconcile_orphaned_running_tasks(tmp_path) == 1
    assert load_task_result(tmp_path, ref)["reason_code"] == "presence_attempt_lost"
    assert len(_room_facts(tmp_path, ref)) == 1


@pytest.mark.parametrize("case", ["live_here", "parked_alive_elsewhere", "no_witness", "local_without_end_fact"])
def test_live_or_unprovable_attempts_keep_being_observed(tmp_path, case):
    # A missing local registry entry is never death (tests/test_presence_continuation_recovery.py).
    ref, stored = _row(tmp_path, f"ev-{case}", recorded=case == "live_here",
                       owner=_witnessed(tmp_path, controller_identity()) if case == "local_without_end_fact" else None,
                       parked=_parked(_alive_witness()) if case == "parked_alive_elsewhere" else None)
    key = (str(tmp_path.resolve()), ref)
    if case == "live_here":
        presence_runner._LIVE_PRESENCE_TASKS.add(key)  # a positive live observation outranks any end fact
    try:
        assert not attempt_ended(stored, tmp_path)
        status, view = retained_work_view(stored, ref, tmp_path)
        assert status == 202 and view["status"] == "pending"
        if case == "parked_alive_elsewhere":
            assert view["outputs"] == [OUTPUT] and view["child_work_ref"] == "child-1"
        assert effective_task_result(tmp_path, stored, materialize_artifacts=False)["status"] == "running"
        assert reconcile_orphaned_running_tasks(tmp_path) == 0
        assert load_task_result(tmp_path, ref)["status"] == "running"
    finally:
        presence_runner._LIVE_PRESENCE_TASKS.discard(key)


@pytest.mark.parametrize("status", ["failed", "completed"])
def test_an_unanswered_terminal_is_never_completed(tmp_path, status):
    ref, stored = _row(tmp_path, f"ev-unknown-{status}", status, unknown=True, answer="salvaged draft")
    code, view = retained_work_view(stored, ref, tmp_path)
    assert code == 200 and view["status"] == "failed" and view["recorded_status"] == status
    assert (view["outcome"], view["text"]) == ("silent", "")
    assert view["uncertainty"]["source"] == "provider_outcome_unknown_no_resend"


def test_a_canonical_answer_keeps_its_projection(tmp_path):
    ref, stored = _row(tmp_path, "ev-answered", "completed", answer="The report is ready.")
    code, view = retained_work_view(stored, ref, tmp_path)
    assert code == 200 and (view["status"], view["outcome"], view["text"]) == (
        "completed", "message", "The report is ready.")
    assert "code" not in view

def test_a_failed_turn_with_a_selected_reply_keeps_it(tmp_path):
    # Task status is not speech: a failed turn that selected a reply still delivers it once.
    ref, stored = _row(tmp_path, "ev-failed-answer", "failed", answer="Partial answer, the rest follows.")
    code, view = retained_work_view(stored, ref, tmp_path)
    assert code == 200 and (view["status"], view["outcome"], view["text"]) == (
        "failed", "message", "Partial answer, the rest follows.")
    assert "code" not in view and view["work_ref"] == ref
