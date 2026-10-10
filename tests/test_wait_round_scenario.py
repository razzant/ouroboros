"""Deterministic BEFORE/AFTER scenario through the real parent loop and wait consumers.

The provider boundary is scripted, not a live model benchmark. Children use real
result/mail writers; the harness lane separately drives real supervised_wait.
Observed input characters are a proxy, NEVER billed/model tokenizer tokens.
Copy this same fixture into a detached base worktree to reproduce BEFORE.
"""
from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from ouroboros import loop
from ouroboros.owner_mailbox import write_task_message
from ouroboros.owner_wait import direct_owner_wait
from ouroboros.task_results import write_task_result
from ouroboros.tools.registry import ToolRegistry
from tests.test_loop_transport_wait import _loop_kwargs


def test_three_native_children_and_one_harness_wait_without_empty_rounds(tmp_path, monkeypatch):
    from ouroboros import owner_wait, task_status
    from ouroboros.delegate_supervision import supervised_wait

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    registry = ToolRegistry(repo_dir=Path(__file__).resolve().parents[1], drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id, ctx.task_attempt = "t-wait", 1
    ctx.owner_wait_callback = direct_owner_wait
    from ouroboros.model_wait import TaskModelWait
    ctx.model_wait_context = TaskModelWait(task={"id": ctx.task_id}, drive_root=tmp_path,
                                           event_queue=None, worker_slot_held=False)
    ctx.model_wait_context.tool_context = ctx
    ctx.task_metadata = {"root_task_id": ctx.task_id}
    write_task_result(tmp_path, ctx.task_id, "running")
    ids = ["native-a", "native-b", "native-c", "harness-d"]
    for tid in ids:
        write_task_result(tmp_path, tid, "running", parent_task_id=ctx.task_id,
                          root_task_id=ctx.task_id, delegation_role="subagent")
    clock = [0.0]
    steps = []
    def signal(text, mid, attention=""):
        args = {"source_task_id": "native-a", "provenance": "descendant_task", "msg_id": mid}
        if attention and "attention_kind" in inspect.signature(write_task_message).parameters:
            args["attention_kind"] = attention
        assert write_task_message(tmp_path, text, ctx.task_id, **args)
    def advance(_seconds):
        clock[0] += 120
        tick = int(clock[0] // 120)
        steps.append(tick)
        if tick in (1, 2, 3):
            signal("informational partial progress", f"info-{tick}")
        elif tick == 4:
            args = {"source_task_id": "parent-context", "provenance": "ancestor_task", "msg_id": "parent-letter"}
            assert write_task_message(tmp_path, "routine parent context", ctx.task_id, **args)
        elif tick == 6:
            signal("question requiring a decision", "question", "question")
        elif tick == 8:
            polls = []
            nanny = SimpleNamespace(task_id="harness-d", drive_root=tmp_path, budget_drive_root=tmp_path,
                task_metadata={"configured_subagent": {"config_fingerprint": "synthetic"}})
            def harness_observation(_ctx, run, _window, _seq):
                polls.append(1)
                return json.dumps({"status": "completed" if len(polls) == 3 else "no_progress",
                                   "run_id": run, "last_seq": len(polls)})
            assert json.loads(supervised_wait(nanny, "run-synthetic", wait_once=harness_observation).text)["status"] == "completed"
            assert len(polls) == 3
            for tid in ids:
                write_task_result(tmp_path, tid, "completed", parent_task_id=ctx.task_id,
                    root_task_id=ctx.task_id, delegation_role="subagent",
                    result=("large final body\n" * 4000 if tid == "native-c" else f"result {tid}"))
        elif tick > 10:
            raise AssertionError("event wait failed to observe terminal facts")
    real_time = task_status.time
    monkeypatch.setattr(task_status, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=advance,
        time=real_time.time))
    monkeypatch.setattr(owner_wait, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=advance,
        time=real_time.time))
    sends = []
    absorbed = [False]
    first_all_receipt = []
    def dispatch(call, _disposition, **_kw):
        # The scripted generation substitutes this transport seam, so retain
        # its normal host observation rather than reusing request one's stamp.
        from ouroboros.loop_delivery import completion_observation
        ctx._completion_observation = completion_observation(ctx, getattr(ctx, "_execution_trace", None) or {})
        sends.append(copy.deepcopy(call.messages))
        # All children were admitted before request 1 (launch boundary 0).
        # Count through the request that FIRST carries every terminal digest,
        # not through completion bookkeeping. Inspect the actual model input,
        # rather than inferring receipt from result-file writes alone.
        for message in call.messages:
            for line in str(message.get("content") or "").splitlines():
                if not line.startswith("{"):
                    continue
                try:
                    payload = json.loads(line)
                except ValueError:
                    continue
                tasks = payload.get("tasks", {})
                if isinstance(tasks, dict) and all(
                    isinstance(tasks.get(tid), dict) and tasks[tid].get("status") == "completed"
                    and len(tasks[tid].get("child_result_sha256", "")) == 64 for tid in ids
                ) and not first_all_receipt:
                    first_all_receipt.append(len(sends))
        assert len(sends) <= 12, json.dumps(sends[-1][-4:], ensure_ascii=False)
        if all(task_status.load_effective_task_result(tmp_path, tid).get("status") == "completed" for tid in ids):
            from ouroboros.tools.join_ledger import _child_result_sha256
            children = [{"child_task_id": tid, "disposition": "integrated", "child_result_sha256":
                _child_result_sha256(task_status.load_effective_task_result(tmp_path, tid))} for tid in ids]
            if absorbed[0]:
                return {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "finish", "type": "function", "function": {"name": "finish_task", "arguments": json.dumps({
                        "action": "finish", "answer": "All four results observed; large detail stays source-linked."})}}]}, 0.0
            absorbed[0] = True
            return {"role": "assistant", "content": None, "tool_calls": [
                {"id": "absorb", "type": "function", "function": {"name": "tree_note", "arguments": json.dumps({
                    "kind": "decision", "text": "Use observed outcomes; retain large source for explicit reading.",
                    "payload": {"type": "child_result_disposition", "children": children}})}}]}, 0.0
        return {"role": "assistant", "content": None, "tool_calls": [{"id": f"wait-{len(sends)}",
                "type": "function", "function": {"name": "wait_tasks", "arguments": json.dumps({"task_ids": ids})}}]}, 0.0
    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    result, _usage, trace = loop.run_llm_loop(**{**_loop_kwargs(tmp_path, registry, []),
        "emit_progress": lambda *_a, **_kw: None})
    assert "All four" in result
    event_default = registry.schemas()[0] is not None and "attention_kind" in inspect.signature(write_task_message).parameters
    report = {"parent_rounds": len(sends), "simulated_seconds": clock[0], "terminal_children": len(ids),
              "launch_parent_round": 0,
              "first_all_results_parent_round": first_all_receipt[0] if first_all_receipt else None,
              "launch_to_all_results_rounds": first_all_receipt[0] if first_all_receipt else None,
              "post_receipt_completion_rounds": len(sends) - first_all_receipt[0] if first_all_receipt else None,
              "input_characters": sum(len(json.dumps(r, ensure_ascii=False)) for r in sends),
              "measurement": "scripted dispatch boundary; characters are not provider tokens",
              "tool_calls": [r["tool"] for r in trace["tool_calls"]],
              "max_wait_body": max(len(m.get("content") or "") for rows in sends for m in rows
                                   if m.get("role") == "tool" or "You slept" in str(m.get("content") or ""))}
    (tmp_path / "wait_scenario.json").write_text(json.dumps(report), encoding="utf-8")
    print("WAIT_SCENARIO " + json.dumps(report))
    if event_default:
        # Request 3 receives all results and absorbs them; request 4 finishes.
        # The TZ bounds launch-through-receipt, not total completion requests.
        assert report["launch_to_all_results_rounds"] == 3
        assert report["post_receipt_completion_rounds"] == 1
        assert len(sends) == 4
        assert report["max_wait_body"] <= 15_000
        assert any("routine parent context" in json.dumps(r) for r in sends[1:])
    else:
        assert len(sends) > 3, "BEFORE must reproduce extra parent requests"
