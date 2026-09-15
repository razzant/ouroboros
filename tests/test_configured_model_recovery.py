"""Supervising cognition recovers without replaying external execution."""

import copy
import hashlib
import json

import httpx
import pytest

from ouroboros import delegate_custody as custody, delegate_hold, loop, loop_transport
from ouroboros import usage_accounting as ua
from ouroboros.delegate_shared import delegate_result
from ouroboros.delegate_start_claims import claimed_start_request
from tests.test_delegate_hold import _configured_registry, _loop_kwargs, _start_leaf
from tests.test_transport_death_retry import _LedgerLLM, _ledger


@pytest.mark.parametrize("external", ["inline", "consumed", "unread", "patch", "pending", "absent", "unreadable"])
def test_saved_external_work_does_not_gate_supervising_cognition(tmp_path, monkeypatch, external):
    """The incident's long consumed result and uncertain custody use one rail.

    Only model I/O is scripted: physical accounting, custody replay, the main
    loop and the fresh-start guard run as production code.
    """
    task_id, run_id = "t-death", "run-completed"
    registry = _configured_registry(tmp_path, task_id)
    output = "External result, including Unicode: готово.\n" * (1000 if external == "consumed" else 1)
    if external in {"inline", "consumed", "unread", "patch"}:
        custody._CUSTODY.pop(run_id, None)
        row = custody.RunCustody(run_id=run_id, task_id=task_id, route_id="claude", model="external")
        if external == "patch":
            row.snapshot_id = "snapshot-preserved"
        assert custody.record_started(tmp_path, row)
        assert custody.settle_run(tmp_path, None, row, {"summary": {
            "state": "succeeded", "spendUsd": 0, "spendEstimated": False,
            "inputTokens": 5, "outputTokens": 5,
        }})["settled"]
        if external in {"consumed", "unread"}:
            data = output.encode("utf-8")
            artifact = tmp_path / "delegated_runs" / f"{run_id}.json"
            artifact.parent.mkdir()
            artifact.write_bytes(data)
            row.output_artifact, row.output_sha, row.output_complete = (
                f"delegated_runs/{run_id}.json", hashlib.sha256(data).hexdigest(), True)
            assert custody.emit(tmp_path, custody.OUTPUT_SPILLED, {
                "run_id": run_id, "task_id": task_id, "artifact": row.output_artifact,
                "sha256": row.output_sha, "bytes": len(data), "staged": True, "full_content": True,
            })
            if external == "consumed":
                assert custody.record_output_consumed(tmp_path, row, artifact=row.output_artifact,
                    byte_length=len(data), sha256=row.output_sha, chars=len(output), lines=1001)
        if external == "patch":
            assert custody.record_patch_captured(tmp_path, row, patch_sha256="preserved-patch")
    if external == "pending":
        assert custody.record_start_requested(tmp_path, task_id=task_id,
            invocation_id="pending-invocation", idempotency_key="pending-invocation", request={"prompt": "already sent"})
    if external == "unreadable":
        monkeypatch.setattr(custody, "custody_log_unreadable", lambda *_: True)

    observations, messages_seen = [], []
    class Model(_LedgerLLM):
        def chat(self, **kwargs):
            messages_seen.append(copy.deepcopy(kwargs["messages"]))
            return super().chat(**kwargs)

    model = Model(tmp_path, lambda: httpx.ReadError("host stream lost after external completion"))
    monkeypatch.setattr(loop_transport, "upstream_transport_reachable",
        lambda *a, **kw: observations.append(model.calls) or {"kind": "upstream_http", "status_code": 200})
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", lambda *a: False)
    monkeypatch.setattr(delegate_hold, "_leaf_probe_live", lambda *a: pytest.fail("no holdable leaf"))
    monkeypatch.setattr(custody, "release_task_runs", lambda *a: None)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    kwargs = _loop_kwargs(tmp_path, registry, [])
    kwargs["llm"] = model
    if external in {"inline", "consumed", "unread", "patch"}:
        kwargs["messages"].extend([
            {"role": "assistant", "content": None, "tool_calls": [{"id": "completed-wait", "type": "function",
                "function": {"name": "delegate_wait", "arguments": json.dumps({"run_id": run_id})}}]},
            {"role": "tool", "tool_call_id": "completed-wait", "content": output},
        ])
    custody_before = custody.event_log_path(tmp_path).read_bytes() if custody.event_log_path(tmp_path).exists() else b""
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id=task_id, root_task_id=task_id, global_limit_usd=100)):
        text, usage, trace = loop.run_llm_loop(**kwargs)

    assert text == "done" and model.calls == 2 and observations == [1]
    assert not trace["tool_calls"]  # No completed delegate/tool execution was replayed.
    assert any("NEW physical model attempt" in str(row.get("content")) for row in messages_seen[-1])
    if external in {"inline", "consumed", "unread", "patch"}:
        assert [row["content"] for row in messages_seen[-1] if row.get("tool_call_id") == "completed-wait"] == [output]
    ledger = _ledger(tmp_path)
    assert [row["state"] for row in ledger] == ["reserved", "dispatched", "unresolved", "reserved", "dispatched", "settled"]
    assert ledger[0]["attempt_id"] != ledger[3]["attempt_id"]
    assert usage["transport_recovery"]["previous_attempt"]["physical_attempt_id"] == ledger[0]["attempt_id"]
    assert ua.usage_projection(tmp_path)["unresolved_upper_bound_usd"] == 1.0
    custody_after = custody.event_log_path(tmp_path).read_bytes() if custody.event_log_path(tmp_path).exists() else b""
    rows = [json.loads(line) for line in custody_after.splitlines()]
    assert [row for row in rows if row.get("type", "").startswith("delegate_")] == [
        json.loads(line) for line in custody_before.splitlines() if json.loads(line).get("type", "").startswith("delegate_")]
    if external in {"pending", "patch", "unreadable"}:
        # Ability to think is not authority to duplicate the external work.
        accepted, refusal = claimed_start_request(tmp_path, claim_target="", payload_busy=lambda *a: "",
            actor_ctx=registry._ctx, enforce_actor_idle=True, task_id=task_id, invocation_id="duplicate")
        assert not accepted
        assert refusal["reason"] == ("replacement_custody_unknown" if external == "unreadable" else "replacement_requires_settlement")


def test_live_hold_takes_over_an_existing_transport_episode(tmp_path, monkeypatch):
    registry = _configured_registry(tmp_path)
    calls, probes, snapshots = [], [], []
    def send(_llm, messages, *args, **kwargs):
        usage = args[8]
        calls.append(len(calls) + 1)
        snapshots.append(copy.deepcopy(messages))
        if len(calls) <= 2:
            if len(calls) == 2:
                # This attempt now owns a live external leaf; the first did not.
                _start_leaf(tmp_path)
            usage.update(_last_llm_error_kind="provider_outcome_unknown",
                         _pending_transport_outcome={"physical_attempt_id": f"old-{len(calls)}"})
            return None, 0.0
        usage.pop("_last_llm_error_kind", None)
        return {"role": "assistant", "content": "integrated"}, 0.0

    def observed(*args, **kwargs):
        probes.append(len(calls))
        assert len(calls) == 1, "A leaf wake must not require a new upstream observation"
        return {"kind": "upstream_http", "status_code": 200}

    monkeypatch.setattr(loop, "call_llm_with_retry", send)
    monkeypatch.setattr(loop_transport, "upstream_transport_reachable", observed)
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", lambda *a: False)
    monkeypatch.setattr(delegate_hold, "_leaf_probe_live", lambda *a: True)
    monkeypatch.setattr(delegate_hold, "supervised_wait", lambda *a: delegate_result({
        "status": "succeeded", "run_id": "run-leaf", "supervision_wake_id": "wake-after-hold"}))
    monkeypatch.setattr(delegate_hold, "acknowledge_pending_wake", lambda *a, **kw: True)
    monkeypatch.setattr(custody, "release_task_runs", lambda *a: None)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    result, usage, _trace = loop.run_llm_loop(**_loop_kwargs(tmp_path, registry, []))
    assert result == "integrated" and len(calls) == 3 and probes == [1]
    assert sum("NEW physical model attempt" in str(row.get("content")) for row in snapshots[-1]) == 1
    assert any("[DELEGATED LEAF WAKE" in str(row.get("content")) for row in snapshots[-1])
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    waits = [row for row in events if row.get("type") == "network_wait"]
    assert [(row["phase"], row.get("detail", "")) for row in waits] == [
        ("entered", ""), ("waiting", ""), ("recovered", "new_attempt_after_unknown_outcome"), ("ended", "hold_latched")]
    assert waits[-1]["outcome_custody"]["physical_attempt_id"] == "old-1"
    assert usage["transport_recovery"]["old_outcome"] == "unknown"
