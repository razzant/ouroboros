"""Real publication seams feed metrics and late tree receipts."""
from types import SimpleNamespace

import pytest

from ouroboros import maintenance, subagent_metrics as metrics
from ouroboros.review_records import ReviewActorRecord, ReviewRequest, ReviewSlot
from ouroboros.review_seat_outcome import finalize_seat_outcome, publish_seat_execution
from ouroboros.subagent_history import record_session_execution, record_task_execution
from ouroboros.task_results import load_task_result, write_task_result
from tests import test_stream_timeout_repeat as stream

wire_env = stream.wire_env


def test_api_history_selects_original_identity_and_observed_call_accounts(tmp_path):
    from ouroboros.utils import append_jsonl
    from ouroboros import store
    from ouroboros.swarm_receipt_sources import _usage
    from ouroboros.usage_store import hold

    task = {"id": "root", "configured_subagent": {"selected_subagent_id": "worker",
        "route": {"kind": "api_model", "target_id": "test/model", "credential_profile_id": "chosen"}}}
    call = {"llm_call_id": "call-one", "model": "test/model", "requested_profile": "chosen",
            "usable_solve_response": True, "response_ref": {"call_id": "response"}, "ts": "2026-10-22T00:01:00Z",
            "observed_route": {"credentialProfileId": "served", "model": "served/model"}}
    with hold(tmp_path) as txn:
        txn.write({"attempt_id": "attempt", "kind": "external_unmetered", "state": "settled",
                   "task_id": "root", "root_task_id": "root", "cost_usd": None, "provider": "test"})
    append_jsonl(tmp_path / "logs/events.jsonl", {"type": "llm_round", "llm_call_id": "call-one",
        "task_id": "root", "ledger_attempt_ids": ["attempt"], "physical_attempt_id": "attempt"})
    record_task_execution(task, {"llm_call_refs": [call, {**call, "llm_call_id": "other", "model": "fallback"}]}, drive_root=tmp_path)
    maintenance.run_job(tmp_path, "subagent_metrics")
    row = metrics.read(tmp_path, "worker")["rows"][0]
    assert row["dispatched"] == row["answer_received"] == 1
    assert row["usable_verdict"] == 0  # execution text does not claim a review verdict
    with store.open_store(tmp_path, write=False) as txn:
        observed = list(txn.set_rows(metrics.set_name("swarm_actors", "root")).values())
    usage, _ = _usage(tmp_path, "root", observed)
    assert usage[0]["actor_id"] == "task:root"
    assert usage[0]["requested_profile"] == "chosen"
    assert usage[0]["observed_route"]["credentialProfileId"] == "served"


@pytest.mark.parametrize("late", [False, True])
def test_api_duration_samples_use_ledger_stamps_not_late_receipt_time(tmp_path, late):
    from ouroboros.usage_store import hold
    from ouroboros.utils import append_jsonl

    previous = None
    steps = [("reserved", "00", {}), ("dispatched", "02", {})]
    if late:
        steps.append(("unresolved", "05", {}))
    steps.append(("settled", "30" if late else "07", {"settle_reason": "late_receipt"} if late else {}))
    for state, seconds, extra in steps:
        with hold(tmp_path) as txn:
            previous = txn.write({"attempt_id": "physical", "kind": "attempt", "state": state,
                "task_id": "task", "root_task_id": "task", "ts": f"2026-10-22T00:00:{seconds}Z", **extra}, previous)
    append_jsonl(tmp_path / "logs/events.jsonl", {"type": "llm_round", "llm_call_id": "call",
        "task_id": "task", "ledger_attempt_ids": ["physical"], "physical_attempt_id": "physical"})
    task = {"id": "task", "configured_subagent": {"selected_subagent_id": "worker",
        "route": {"kind": "api_model", "target_id": "test/model"}}}
    record_task_execution(task, {"llm_call_refs": [{"llm_call_id": "call", "model": "test/model",
        "usable_solve_response": True, "response_ref": {"call_id": "response"}}]}, drive_root=tmp_path)
    maintenance.run_job(tmp_path, "subagent_metrics")
    row = metrics.read(tmp_path, "worker")["rows"][0]
    assert row["duration_samples"] == ({"count": 0, "sum_sec": 0} if late else {"count": 1, "sum_sec": 5})
    assert row["queue"]["fact"]["sum_sec"] == 2
    assert row["duration_basis"] == "sum_of_physical_dispatch_to_settlement_seconds"


@pytest.mark.parametrize("received", [True, False])
def test_real_api_error_producer_separates_received_output_from_usability(wire_env, received):
    from ouroboros import store

    prefix = stream.OUTPUT["content"] if received else []
    client = stream.WireLLM(stream.Wire(*prefix, stream.error_frame(400, "invalid_request")))
    usage = {}
    reply, _ = stream.round_call(client, wire_env, usage, child=False)
    assert reply is None and len(client.sent) == 1
    call = usage["llm_call_refs"][0]
    # The error producer retains positive output evidence, but currently omits
    # the decoder's False. An omitted receipt must remain unknown downstream.
    expected = True if received else None
    assert call.get("model_output_received") is expected and not call.get("usable_solve_response")
    task = {"id": stream.CHILD, "configured_subagent": {"selected_subagent_id": "worker",
        "route": {"kind": "api_model", "target_id": stream.MODEL}}}
    record_task_execution(task, usage, drive_root=wire_env)
    maintenance.run_job(wire_env, "subagent_metrics")
    with store.open_store(wire_env, write=False) as txn:
        producer = txn.get_row(metrics.PRODUCERS, "call:" + call["llm_call_id"])
    assert producer["answer_received"] is expected and producer["usable_solve_response"] is False
    row = metrics.read(wire_env, "worker")["rows"][0]
    assert row["answer_received"] == int(received) and row["no_answer"] == 0


@pytest.mark.parametrize("received", [False, None])
def test_api_history_distinguishes_proven_silence_from_absent_receipt(tmp_path, received):
    from ouroboros import store

    task = {"id": "task", "configured_subagent": {"selected_subagent_id": "worker",
        "route": {"kind": "api_model", "target_id": "test/model"}}}
    call = {"llm_call_id": "call", "model": "test/model", "dispatched": True, "failure_code": "provider_error"}
    if received is not None:
        call["model_output_received"] = received
    record_task_execution(task, {"llm_call_refs": [call]}, drive_root=tmp_path)
    with store.open_store(tmp_path, write=False) as txn:
        producer = txn.get_row(metrics.PRODUCERS, "call:call")
    assert producer["answer_received"] is received and producer["usable_solve_response"] is False
    maintenance.run_job(tmp_path, "subagent_metrics")
    row = metrics.read(tmp_path, "worker")["rows"][0]
    assert row["answer_received"] == 0 and row["no_answer"] == int(received is False)


@pytest.mark.parametrize("failure,terminal,interrupted", [
    ("provider_error", True, False), ("provider_outcome_unknown", False, False), ("interrupted", True, True),
])
def test_absent_api_receipt_fact_stays_unknown_through_pending_and_interruption(tmp_path, failure, terminal, interrupted):
    from ouroboros import store

    task = {"id": "task", "configured_subagent": {"selected_subagent_id": "worker",
        "route": {"kind": "api_model", "target_id": "test/model"}}}
    record_task_execution(task, {"llm_call_refs": [{"llm_call_id": "call", "model": "test/model",
        "dispatched": True, "failure_code": failure}]}, drive_root=tmp_path)
    with store.open_store(tmp_path, write=False) as txn:
        producer = txn.get_row(metrics.PRODUCERS, "call:call")
    assert producer["answer_received"] is None and producer["terminal"] is terminal
    maintenance.run_job(tmp_path, "subagent_metrics")
    row = metrics.read(tmp_path, "worker")["rows"][0]
    assert row["answer_received"] == row["no_answer"] == 0
    assert row["pending"] == int(not terminal) and row["interrupted"] == int(interrupted)


@pytest.mark.parametrize("text,usable", [('[]', True), ('malformed answer', False)])
def test_canonical_review_publication_separates_received_and_usable(tmp_path, monkeypatch, text, usable):
    from ouroboros.swarm_receipt import publish

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    write_task_result(tmp_path, "root", "completed")
    publish(tmp_path, "root")
    before = load_task_result(tmp_path, "root")["swarm_receipt"]
    request = ReviewRequest(surface="result_review", goal="review", task_id="root", policy={"parts": ["result"]})
    slot = ReviewSlot(slot_id="worker", model="test/model", subagent_id="worker")
    actor = ReviewActorRecord(slot_id="worker", model="test/model", status="ok", raw_text=text,
                              operation_id="operation", duration_sec=1)
    finalize_seat_outcome(request, slot, actor)
    ctx = SimpleNamespace(drive_root=tmp_path, budget_drive_root=tmp_path)
    publish_seat_execution(request, slot, actor, ctx)
    maintenance.run_job(tmp_path, "subagent_metrics")
    row = metrics.read(tmp_path, "worker")["rows"][0]
    assert row["answer_received"] == 1
    assert row["usable_verdict"] == int(usable)
    assert load_task_result(tmp_path, "root")["swarm_receipt"]["revision"] > before["revision"]
    publish_seat_execution(request, slot, actor, ctx)
    assert metrics.read(tmp_path, "worker")["rows"][0]["observations"] == 1


@pytest.mark.parametrize("operation", ["", "immutable-operation"])
def test_review_without_a_producer_identity_does_not_invent_one(tmp_path, monkeypatch, operation):
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    request = ReviewRequest(surface="result_review", goal="review", task_id="task", policy={"parts": ["result"]})
    slot = ReviewSlot(slot_id="worker", model="test/model")
    actor = ReviewActorRecord(slot_id="worker", model="test/model", status="ok", raw_text="[]", operation_id=operation)
    finalize_seat_outcome(request, slot, actor)
    publish_seat_execution(request, slot, actor, SimpleNamespace(drive_root=tmp_path, budget_drive_root=tmp_path))
    assert len(metrics.read(tmp_path, "worker")["rows"]) == int(bool(operation))


def test_session_plan_and_execution_collection_keep_identity_and_nanny_separate(tmp_path):
    from ouroboros.delegate_custody import RunCustody
    from ouroboros.swarm_receipt import publish

    write_task_result(tmp_path, "root", "completed", supervisor_route={"model": "paid/api", "cost_kind": "unknown"})
    publish(tmp_path, "root")
    for mode in ("plan", "execution"):
        custody = RunCustody(task_id="root", run_id=mode, route_id="codex", model="leaf", selected_subagent_id="worker",
                             mode=mode, started_ts="2026-10-22T00:00:00Z")
        record_session_execution(tmp_path, custody, {"summary": {"state": "succeeded", "answer_received": True,
            "startedAt": "2026-10-22T00:00:00Z", "finishedAt": "2026-10-22T00:00:05Z", "durationSec": 5}},
                                 {"model": "served", "profile_id": "subscription-account"})
    maintenance.run_job(tmp_path, "subagent_metrics")
    rows = metrics.read(tmp_path, "worker")["rows"]
    assert {row["purpose"] for row in rows} == {"plan", "execution"}
    assert all(row["nanny_billing"]["fact"]["model"] == "paid/api" for row in rows)
    result = load_task_result(tmp_path, "root")["swarm_receipt"]
    assert {actor["id"] for actor in result["actor_intervals"]} >= {"run:plan", "run:execution"}


def test_review_wait_records_return_but_not_a_false_selected_terminal(tmp_path):
    from ouroboros import store
    from ouroboros.review_pause import receptive_author

    for interrupted in (False, True):
        binding = {"drive_root": tmp_path, "task_id": "root", "selection": {"tasks": ["child"]}}
        try:
            with receptive_author(binding):
                if interrupted:
                    raise ValueError("interrupted fixture")
        except ValueError:
            pass
    with store.open_store(tmp_path, write=False) as txn:
        waits = txn.set_rows(metrics.set_name("swarm_waits", "root"))
    assert {row["woke_by"] for row in waits.values()} == {"whole_panel_returned", "unwound"}
    assert all(row["selection"]["tasks"] == ["child"] for row in waits.values())
