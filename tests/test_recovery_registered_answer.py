"""Recovery never re-runs an answered prompt (Astra R2, consilium unanimous).

With the answer on the live bus BEFORE the result store (owner 1A(1)), a worker
that dies or stalls inside the store leaves a task whose row is not terminal
while the host already holds its answer (owed in the outbox, delivered with a
receipt, or unreadable evidence). Both recovery owners — the hard-timeout reaper
and dead-worker recovery — finalize that attempt from the outbox's bytes through
the lifecycle-fault vocabulary instead of a fresh model attempt over the original
prompt. Without a registered answer each keeps its ordinary path (the guard is
two-sided): the reaper's timeout row and salvage, the crash retry.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros.task_results import load_task_result, write_task_result
from supervisor import terminal_delivery as td
from tests._cancel_intents_shared import qenv as _qenv

# The fixture is requested by name as a test parameter, so it is re-bound through a
# module attribute (a direct import that reappears as a parameter is an F811 under ruff).
qenv = _qenv


def _event(tid: str, text: str = "the authored answer") -> dict:
    return {"type": "send_message", "task_id": tid, "chat_id": 1, "text": text,
            "delivery_id": td.delivery_id_for(tid, text), "terminal_origin": "model_final"}


def _task(tid: str) -> dict:
    return {"id": tid, "type": "task", "chat_id": 1, "depth": 0, "root_task_id": tid,
            "parent_task_id": "", "delegation_role": "root", "_attempt": 1}


def _register(drive, tid: str, state: str) -> dict:
    evt = _event(tid)
    if state == "unreadable":
        registry = drive / "state" / "terminal_deliveries.json"
        registry.parent.mkdir(parents=True, exist_ok=True)
        registry.write_text("{bad json", encoding="utf-8")
        return evt
    assert td.register_pending_delivery(drive, evt)
    if state == "delivered":
        assert td.register_delivery(drive, evt["delivery_id"], emitted=evt)
    if state == "exhausted":  # the host gave up: the row left pending, payload and copy kept
        (owed,) = td.pending_deliveries(drive)
        td._retire_exhausted(drive, [owed])
        assert td.terminal_answer_receipts(drive, tid)["state"] == "exhausted"
    return evt


@pytest.mark.parametrize("state", ["owed", "exhausted", "delivered", "unreadable"])
def test_a_hard_timeout_finalizes_a_registered_answer_instead_of_rerunning_the_prompt(qenv, monkeypatch, state):
    """The reaper's own hard-deadline job (``queue_timeouts`` shape) over an attempt whose answer the host holds."""
    from ouroboros.headless import prepare_terminal_task_files
    from ouroboros.working_checkpoint import REASON_FINALIZATION_INCOMPLETE, final_answer_registered
    from supervisor import task_reaper as reaper, worker_health, workers

    tid = "early-answer-crash"
    write_task_result(qenv.drive, tid, "running", result="working", task_attempt=1)
    evt = _register(qenv.drive, tid, state)
    assert final_answer_registered(qenv.drive, tid)
    assert prepare_terminal_task_files(qenv.drive, _task(tid))["terminal_source_present"] is False
    monkeypatch.setattr(reaper, "_kill_and_confirm_worker_dead", lambda *a: True)
    monkeypatch.setattr(workers, "_reconcile_confirmed_dead_review_owner", lambda *a: None)
    monkeypatch.setattr(worker_health, "_retire_dead_model_consumers", lambda *a, **k: None)
    monkeypatch.setattr("ouroboros.tools.services.archive_task_service_logs", lambda *a: None)
    monkeypatch.setattr("supervisor.cancel_publication._audit_delegated_runs_on_kill", lambda *a, **k: {})
    monkeypatch.setattr(qenv.q, "reconstruct_task_cost", lambda *a, **k: {})
    monkeypatch.setattr(reaper, "send_with_budget", lambda *a, **k: pytest.fail("an adopted answer gets no stop notice"))
    monkeypatch.setattr(reaper, "_deliver_reap_salvage", lambda *a: pytest.fail("no second salvage"))
    monkeypatch.setattr(reaper, "_emit_reap_task_done", lambda *a, **k: pytest.fail("published as its own terminal"))
    finished = []
    monkeypatch.setattr(reaper, "_finish_self_finalized_task", lambda *a, **k: finished.append(a[5]))

    reaper.reap_timed_out_task({"worker_id": 0, "worker": None, "proc": SimpleNamespace(pid=0),
                                "drive_root": str(qenv.drive), "task_id": tid, "task": _task(tid),
                                "task_type": "task", "attempt": 1, "terminal_reason": "deadline",
                                "runtime_sec": 300, "deadline_reached": True, "ceiling_reached": False})

    assert finished == ["failed"] and qenv.q.PENDING == [], "the original prompt was placed in PENDING"
    stored = load_task_result(qenv.drive, tid)
    assert stored["status"] == "failed" and stored["reason_code"] == REASON_FINALIZATION_INCOMPLETE
    assert stored["outcome_axes"]["execution"]["status"] == "infra_failed"
    assert stored["finalization_recovery"] == {"cause": "deadline", "answer": state,
                                               "delivery_id": "" if state == "unreadable" else evt["delivery_id"],
                                               "attempt": 1}
    assert "not run again" in stored["terminal_host_notice"]
    if state == "unreadable":
        assert "could not be read" in stored["terminal_host_notice"]
        assert stored["result"] == stored["terminal_host_notice"] and "terminal_origin" not in stored
    else:
        # The outbox's exact bytes (the pending row, or the retained receipt) become the result.
        assert stored["result"] == "the authored answer" and stored["terminal_origin"] == "model_final"
        assert {"owed": "the outbox delivers it", "exhausted": "was given up",
                "delivered": "delivered"}[state] in stored["terminal_host_notice"]
        if state == "exhausted":  # never the owed sentence once the host stopped sending
            assert "outbox delivers" not in stored["terminal_host_notice"]
            assert "its copy is kept at" in stored["terminal_host_notice"]
    assert final_answer_registered(qenv.drive, tid), "finalizing never retires the answer's custody"


@pytest.mark.parametrize("state", ["owed", "delivered", "unreadable"])
@pytest.mark.parametrize("exitcode", [1, -11])
def test_dead_worker_recovery_finalizes_a_registered_answer_without_a_requeue(qenv, monkeypatch, state, exitcode):
    from supervisor import task_reaper, worker_health as health, workers

    tid = "crash-after-answer"
    task = _task(tid)
    write_task_result(qenv.drive, tid, "running", result="working", task_attempt=1)
    evt = _register(qenv.drive, tid, state)
    monkeypatch.setattr(workers, "DRIVE_ROOT", qenv.drive)
    monkeypatch.setattr(workers, "reconstruct_task_cost",
                        lambda *_a, **_k: {"cost_accounting_status": "unavailable", "cost_final": False})
    monkeypatch.setattr(health, "_dead_job_is_current", lambda job: True)
    monkeypatch.setattr(health, "_complete_exact_budget_pause_after_death", lambda *_a, **_k: (False, False))
    finished: list = []
    monkeypatch.setattr(task_reaper, "_finish_self_finalized_task",
                        lambda *args, **kwargs: finished.append((args, kwargs)))
    monkeypatch.setattr(qenv.q, "enqueue_task",
                        lambda *_a, **_k: pytest.fail("the original prompt must not be requeued"))
    sent: list = []
    monkeypatch.setattr(workers, "send_with_budget", lambda *args, **kwargs: sent.append(args), raising=False)
    worker = SimpleNamespace(proc=SimpleNamespace(pid=4321, is_alive=lambda: False), active_capacity=True)
    job = {"worker": worker, "task_id": tid, "task": task, "meta": {"task": task, "attempt": 1},
           "worker_id": 0, "exitcode": exitcode, "drive_root": str(qenv.drive)}

    health._recover_crashed_task_without_terminal(job, qenv.q)

    (args, kwargs), = finished
    assert args[3] == tid and args[5] == "failed" and kwargs == {"worker_id": 0}
    stored = load_task_result(qenv.drive, tid)
    assert args[6]["status"] == stored["status"] == "failed"
    assert stored["reason_code"] == "finalization_incomplete"
    assert stored["finalization_recovery"]["cause"] == "worker_crash"
    if state != "unreadable":
        assert stored["result"] == "the authored answer"
        assert stored["finalization_recovery"]["delivery_id"] == evt["delivery_id"]
    assert sent == [], "no crash toast: the answer and the honest card speak"


def test_a_refused_finalization_write_keeps_the_job_for_the_next_health_tick(qenv, monkeypatch):
    """A registered answer with a store that refuses the write: no retry, no silent drop."""
    from supervisor import task_reaper, worker_health as health, workers

    tid = "store-refuses"
    task = _task(tid)
    write_task_result(qenv.drive, tid, "running", result="working", task_attempt=1)
    _register(qenv.drive, tid, "owed")
    monkeypatch.setattr(workers, "DRIVE_ROOT", qenv.drive)
    monkeypatch.setattr(workers, "reconstruct_task_cost",
                        lambda *_a, **_k: {"cost_accounting_status": "unavailable", "cost_final": False})
    monkeypatch.setattr(health, "_dead_job_is_current", lambda job: True)
    monkeypatch.setattr(health, "_complete_exact_budget_pause_after_death", lambda *_a, **_k: (False, False))
    monkeypatch.setattr(qenv.q, "enqueue_task",
                        lambda *_a, **_k: pytest.fail("the original prompt must not be requeued"))

    def refuse(*_args, **_kwargs):
        raise OSError("store unavailable")

    monkeypatch.setattr("ouroboros.task_results.write_task_result", refuse)
    worker = SimpleNamespace(proc=SimpleNamespace(pid=4321, is_alive=lambda: False), active_capacity=True)
    job = {"worker": worker, "task_id": tid, "task": task, "meta": {"task": task, "attempt": 1},
           "worker_id": 0, "exitcode": 1, "drive_root": str(qenv.drive)}
    with pytest.raises(task_reaper.TerminalFileRecoveryPending):
        health._recover_crashed_task_without_terminal(job, qenv.q)
    assert load_task_result(qenv.drive, tid)["status"] == "running"


def test_registered_answer_reads_the_outbox_never_the_result_prose(tmp_path):
    from ouroboros.working_checkpoint import registered_answer

    tid = "custody-read"
    write_task_result(tmp_path, tid, "running", result="prose the host must not adopt")
    assert registered_answer(tmp_path, tid) == {}
    evt = _event(tid, "owed bytes")
    assert td.register_pending_delivery(tmp_path, evt)
    assert registered_answer(tmp_path, tid) == {"basis": "owed", "text": "owed bytes", "delivery_id": evt["delivery_id"],
                                                "terminal_origin": "model_final"}
    assert td.register_delivery(tmp_path, evt["delivery_id"], emitted=evt)
    assert registered_answer(tmp_path, tid) == {"basis": "delivered", "text": "owed bytes",
                                                "delivery_id": evt["delivery_id"], "terminal_origin": "model_final"}
    # A legacy delivered id proves no bytes; it still refuses regeneration.
    legacy = td.delivery_id_for(tid, "legacy")
    assert td.register_delivery(tmp_path, legacy)
    registry = tmp_path / "state" / "terminal_deliveries.json"
    data = json.loads(registry.read_text(encoding="utf-8"))
    data["receipts"].pop(evt["delivery_id"])
    registry.write_text(json.dumps(data), encoding="utf-8")
    assert registered_answer(tmp_path, tid)["basis"] == "delivered"
    assert registered_answer(tmp_path, tid)["text"] == ""


@pytest.mark.parametrize("state", ["owed", "delivered", "none"])
def test_a_final_timeout_without_retry_also_finalizes_a_registered_answer(qenv, monkeypatch, state):
    """A hard deadline or ceiling adopts the held answer; without a registered answer the
    ordinary timeout row and salvage still happen. Either way the dead attempt's consumers retire,
    its controls reset, and nothing is queued after it."""
    from supervisor import task_reaper as reaper, worker_health, workers

    tid = "answer-at-deadline"
    task = {**_task(tid), "_attempt": 1}
    write_task_result(qenv.drive, tid, "running", result="working", task_attempt=1)
    if state != "none":
        evt = _event(tid, "complete authored answer")
        assert td.register_pending_delivery(qenv.drive, evt)
        if state == "delivered":
            assert td.register_delivery(qenv.drive, evt["delivery_id"], emitted=evt)
    monkeypatch.setattr(reaper, "_kill_and_confirm_worker_dead", lambda *a: True)
    monkeypatch.setattr(workers, "_reconcile_confirmed_dead_review_owner", lambda *a: None)
    retired, reset = [], []
    monkeypatch.setattr(worker_health, "_retire_dead_model_consumers", lambda job: retired.append(job["task_id"]))
    monkeypatch.setattr("ouroboros.tools.services.archive_task_service_logs", lambda *a: None)
    monkeypatch.setattr("supervisor.cancel_publication._audit_delegated_runs_on_kill", lambda *a, **k: {})
    monkeypatch.setattr(qenv.q, "reconstruct_task_cost", lambda *a, **k: {})
    monkeypatch.setattr("ouroboros.owner_hurry.retry_reset",
                        lambda _drive, _root, task_id, **_k: reset.append(task_id))
    monkeypatch.setattr(reaper, "send_with_budget", lambda *a, **k: None)
    salvage, finished = [], []
    monkeypatch.setattr(reaper, "_deliver_reap_salvage", lambda *a: salvage.append(a))
    monkeypatch.setattr(reaper, "_emit_reap_task_done", lambda *a, **k: None)
    monkeypatch.setattr(reaper, "_finish_self_finalized_task", lambda *a, **k: finished.append(a[5]))
    reaper.reap_timed_out_task({"worker_id": 0, "worker": None, "proc": SimpleNamespace(pid=0),
                                "drive_root": str(qenv.drive), "task_id": tid, "task": task, "task_type": "task",
                                "attempt": 1, "terminal_reason": "hard_deadline", "runtime_sec": 300,
                                "deadline_reached": True})
    stored = load_task_result(qenv.drive, tid)
    assert retired == reset == [tid] and qenv.q.PENDING == [], "no successor follows a hard bound"
    if state == "none":
        assert stored["reason_code"] == "hard_deadline" and len(salvage) == 1 and finished == []
    else:
        assert (stored["result"], stored["reason_code"]) == ("complete authored answer", "finalization_incomplete")
        assert salvage == [] and finished == ["failed"], "published as a terminal, never a second salvage message"


@pytest.mark.parametrize("answer_state", ["owed", "delivered"])
@pytest.mark.parametrize("origin", ["host_notice", "host_salvage"])
def test_recovery_keeps_the_host_terminal_provenance(tmp_path, answer_state, origin):
    """A host-written final (provider-failure notice or salvage receipt) stays host-authored after
    recovery; a salvage receipt yields the full preserved bytes and their path, as a store would."""
    from ouroboros.task_finalization import prepare_terminal_send_event, register_final_answer_owed
    from ouroboros.working_checkpoint import adopt_registered_answer

    tid = "host-authored-terminal"
    task = {**_task(tid), "chat_id": 1}
    text = "The provider is unavailable; here is the preserved intermediate work."
    write_task_result(tmp_path, tid, "running", result="working", task_attempt=1)
    usage = {"terminal_origin": origin}
    event = prepare_terminal_send_event(tmp_path, task, text, usage,
                                        {"type": "send_message", "task_id": tid, "chat_id": 1, "text": text},
                                        presence=False)
    assert event["terminal_origin"] == origin and event["role"] == "system"
    assert register_final_answer_owed(task, event, env_drive_root=tmp_path) == "registered"
    if answer_state == "delivered":
        assert td.register_delivery(tmp_path, event["delivery_id"], emitted=event)
    stored = adopt_registered_answer(tmp_path, task, tid, attempt=1, cause="worker_crash")
    assert stored["terminal_origin"] == origin, "host text is never relabelled as the model's final"
    if origin == "host_salvage":
        assert stored["result"] == text and stored["terminal_salvage_path"] == usage["terminal_salvage_path"]
    else:
        assert stored["result"] == event["text"] and "terminal_salvage_path" not in stored


def test_an_unstamped_legacy_answer_claims_no_origin(tmp_path):
    from ouroboros.working_checkpoint import adopt_registered_answer

    tid = "legacy-final"
    write_task_result(tmp_path, tid, "running", result="working", task_attempt=1)
    legacy = {"type": "send_message", "task_id": tid, "chat_id": 1, "text": "old bytes",
              "delivery_id": td.delivery_id_for(tid, "old bytes")}
    assert td.register_pending_delivery(tmp_path, legacy)
    stored = adopt_registered_answer(tmp_path, _task(tid), tid, attempt=1, cause="worker_crash")
    assert stored["result"] == "old bytes" and "terminal_origin" not in stored


@pytest.mark.parametrize("answer_state", ["owed", "delivered"])
def test_a_timeout_keeps_the_registered_host_salvage_not_a_later_intermediate(qenv, monkeypatch, answer_state):
    """The reaper adopts a held answer BEFORE its history salvage: a later long intermediate response
    must not replace the preserved copy a registered host salvage points at (both are salvaged/<task>.txt)."""
    from pathlib import Path

    from ouroboros import observability
    from ouroboros.task_finalization import prepare_terminal_send_event, register_final_answer_owed
    from supervisor import task_reaper as reaper, worker_health, workers
    from tests.test_delivery_forced_finalization import _forced_test_context

    loop, registry, ctx, trace = _forced_test_context(qenv.drive)
    tid = "parent1"
    task = {**_task(tid), "_attempt": 1}
    answer = "Complete useful answer retained before the provider failed."
    loop._replace_delivery_candidate(registry, ctx, trace, answer, control="replace")
    registry._ctx._owner_directives = [{"content": "Late answer constraint"}]
    latest = "I will revise the answer after checking the new constraint.\n" + "Intermediate notes. " * 400
    observability.persist_call(qenv.drive, task_id=tid, call_id="llm_2_response",
                               call_type="llm_response", payload={"message": {"content": latest}})
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *a, **k: (None, 0.0))
    text, usage, _ = loop._handle_provider_unavailable(ctx)
    assert text == answer and usage["terminal_origin"] == "host_salvage"
    write_task_result(qenv.drive, tid, "running", result="working", task_attempt=1)
    event = prepare_terminal_send_event(qenv.drive, task, text, usage,
                                        {"type": "send_message", "task_id": tid, "chat_id": 1, "text": text},
                                        presence=False)
    assert register_final_answer_owed(task, event, env_drive_root=qenv.drive) == "registered"
    if answer_state == "delivered":
        assert td.register_delivery(qenv.drive, event["delivery_id"], emitted=event)
    preserved = Path(usage["terminal_salvage_path"])
    monkeypatch.setattr(reaper, "_kill_and_confirm_worker_dead", lambda *a: True)
    monkeypatch.setattr(workers, "_reconcile_confirmed_dead_review_owner", lambda *a: None)
    monkeypatch.setattr(worker_health, "_retire_dead_model_consumers", lambda *a, **k: None)
    monkeypatch.setattr("ouroboros.tools.services.archive_task_service_logs", lambda *a: None)
    monkeypatch.setattr("supervisor.cancel_publication._audit_delegated_runs_on_kill", lambda *a, **k: {})
    monkeypatch.setattr(qenv.q, "reconstruct_task_cost", lambda *a, **k: {})
    monkeypatch.setattr("ouroboros.owner_hurry.retry_reset", lambda *a, **k: None)
    monkeypatch.setattr(reaper, "send_with_budget", lambda *a, **k: None)
    monkeypatch.setattr(reaper, "_deliver_reap_salvage", lambda *a: pytest.fail("no second salvage"))
    monkeypatch.setattr(reaper, "_emit_reap_task_done", lambda *a, **k: None)
    monkeypatch.setattr(reaper, "_finish_self_finalized_task", lambda *a, **k: None)
    reaper.reap_timed_out_task({"worker_id": 0, "worker": None, "proc": SimpleNamespace(pid=0),
                                "drive_root": str(qenv.drive), "task_id": tid, "task": task, "task_type": "task",
                                "attempt": 1, "terminal_reason": "hard_deadline", "runtime_sec": 300,
                                "deadline_reached": True})
    stored = load_task_result(qenv.drive, tid)
    assert stored["terminal_origin"] == "host_salvage" and stored["reason_code"] == "finalization_incomplete"
    assert stored["result"] == answer, "the history salvage replaced the registered answer's preserved bytes"
    assert preserved.read_text(encoding="utf-8") == answer and qenv.q.PENDING == []
