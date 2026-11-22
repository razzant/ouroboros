from __future__ import annotations

import json


def write_child(
    tmp_path,
    child_id: str = "child1",
    status: str = "completed",
    **fields,
):
    """Write the canonical child-result fixture shared by delivery tests."""
    from ouroboros.task_results import write_task_result

    return write_task_result(
        tmp_path,
        child_id,
        status,
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="subagent",
        result="full child result",
        trace_summary="trace",
        artifact_status="ready",
        artifacts=[{"kind": "report", "name": "report.md", "sha256": "a" * 64}],
        **fields,
    )


def write_confirmed_disposition_fixture(
    tmp_path,
    *,
    child_id: str = "child1",
    disposition: str,
    rationale: str,
):
    """Append the authoritative task-tree row used by loop/outcome unit tests."""

    from ouroboros.task_status import load_effective_task_result
    from ouroboros.task_tree_ledger import tree_ledger_append
    from ouroboros.tools.join_ledger import _child_result_sha256

    child = load_effective_task_result(tmp_path, child_id)
    result_sha256 = _child_result_sha256(child)
    root_task_id = str(child.get("root_task_id") or "parent1")
    parent_task_id = str(child.get("parent_task_id") or "parent1")
    written = tree_ledger_append(
        root_task_id,
        "decision",
        rationale,
        task_id=parent_task_id,
        role="orchestrator",
        payload={
            "type": "child_result_disposition",
            "child_task_id": child_id,
            "disposition": disposition,
            "child_result_sha256": result_sha256,
        },
        allow_child_result_disposition=True,
        data_root=tmp_path,
    )
    assert written.startswith("OK:")
    return load_effective_task_result(tmp_path, child_id)


def holds(root):
    """The completion_held rows of the events log under ``root`` (absent file: none)."""
    path = root / "logs" / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return [row for row in rows if row.get("type") == "task_checkpoint" and row.get("checkpoint_kind") == "completion_held"]


def _run_loop(
    tmp_path,
    monkeypatch,
    responses,
    review_holds=None,
    *,
    child=False,
    bind_child_before_second=False,
    progress=None,
    setup_context=None,
):
    import queue

    import ouroboros.loop as loop
    from ouroboros.tools.registry import ToolRegistry

    tmp_path.mkdir(parents=True, exist_ok=True)
    if child:
        write_child(tmp_path)
    answers = iter(responses)
    calls = []
    progress = progress if progress is not None else []

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call(_llm, request_messages, *_args, **_kwargs):
        calls.append([dict(row) for row in request_messages])
        if child and bind_child_before_second and len(calls) == 2:
            write_confirmed_disposition_fixture(
                tmp_path,
                disposition="integrated",
                rationale="test parent consumed the handoff",
            )
        answer = next(answers)
        if isinstance(answer, dict):
            return {"role": "assistant", **answer}, 0.0
        return {"role": "assistant", "content": answer}, 0.0

    holds = iter(review_holds or [])

    def fake_review_at_delivery(ctx, _content, _llm_trace, hold):
        # The result review's host step at delivery: a blocking hold shows its outcome to the
        # author and retains the first answer, so the next complete answer is selected
        # explicitly through the real completion door (a non-blocking outcome shown once leaves
        # the next prose fresh: tests/test_completion_hold_observation.py).
        held = next(holds, False)
        if held:
            ctx._completion_hold_cause = "blocking_refusal"
            hold("[Result review] The review's findings were shown before delivery.")
        return held

    monkeypatch.setattr(loop, "call_llm_with_retry", fake_call)
    monkeypatch.setattr("ouroboros.tools.review_result.result_review_at_delivery", fake_review_at_delivery)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_metadata = {
        "budget_drive_root": str(tmp_path),
        "root_task_id": "parent1",
    }
    if setup_context:
        setup_context(registry._ctx)
    result, usage, trace = loop.run_llm_loop(
        messages=[{"role": "user", "content": "do the work"}],
        tools=registry,
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda text, **facts: progress.append({"text": text, **facts}),
        incoming_messages=queue.Queue(),
        task_id="parent1",
        drive_root=tmp_path,
    )
    return result, usage, trace, calls
