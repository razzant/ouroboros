"""``_store_task_result`` persistence semantics.

Split out of ``tests/test_agent_task_pipeline.py`` when that module was divided
by theme; every moved block is verbatim. Covers review-evidence persistence,
the compact review projection (no raw model text), failed-status preservation,
and the unresolved-vs-recovered tool-failure outcome axes.
"""

import json
from types import SimpleNamespace

import ouroboros.agent_task_pipeline as pipeline
from tests._store_shared import read_result


def test_store_task_result_persists_review_evidence(tmp_path):
    env = SimpleNamespace(drive_root=tmp_path)

    pipeline._store_task_result(
        env=env,
        task={"id": "task-store", "type": "task", "text": "hi"},
        text="done",
        usage={"rounds": 2, "cost": 0.1},
        llm_trace={"tool_calls": [], "reasoning_notes": []},
        review_evidence={"has_evidence": True, "open_obligations": [{"item": "tests_affected"}]},
    )

    payload = read_result(tmp_path, "task-store")
    assert payload["review_evidence"]["has_evidence"] is True
    assert payload["review_evidence"]["open_obligations"][0]["item"] == "tests_affected"


def test_store_task_result_persists_only_compact_review_projection(tmp_path):
    env = SimpleNamespace(drive_root=tmp_path)
    trace = {
        "tool_calls": [],
        "review_runs": [{
            "request": {"surface": "task_acceptance", "policy": {"min_successful_slots": 1}},
            "authority": "host_root",
            "aggregate_signal": "DEGRADED",
            "actors": [{
                "slot_id": "slot_1", "model": "openai/gpt-5.6-sol", "status": "ok",
                "parsed": {"verdict": "DEGRADED", "summary": "not enough evidence"},
                "signal": "DEGRADED", "raw_text": "PRIVATE RAW MODEL RESPONSE",
            }],
        }],
    }
    pipeline._store_task_result(
        env=env,
        task={"id": "task-review-projection", "type": "task", "text": "hi"},
        text="done",
        usage={"rounds": 1, "cost": 0.0},
        llm_trace=trace,
        review_evidence={},
    )

    payload = read_result(tmp_path, "task-review-projection")
    actor = payload["review_projection"]["panels"][0]["actors"][0]
    assert actor["model"] == "openai/gpt-5.6-sol"
    assert actor["parse_status"] == "valid"
    assert actor["semantic_verdict"] == "DEGRADED"
    assert "raw_text" not in actor
    assert "PRIVATE RAW MODEL RESPONSE" not in json.dumps(payload)


def test_store_task_result_preserves_failed_status(tmp_path):
    from ouroboros.task_results import STATUS_FAILED, write_task_result

    env = SimpleNamespace(drive_root=tmp_path)
    write_task_result(tmp_path, "task-failed", STATUS_FAILED, result="initial failure")

    pipeline._store_task_result(
        env=env,
        task={"id": "task-failed", "type": "task", "text": "hi"},
        text="final failure reply",
        usage={"rounds": 1, "cost": 0.0},
        llm_trace={"tool_calls": [], "reasoning_notes": []},
        review_evidence={},
    )

    payload = read_result(tmp_path, "task-failed")
    assert payload["status"] == STATUS_FAILED
    assert payload["result"] == "final failure reply"


def test_store_task_result_keeps_delivered_answer_and_unresolved_tool_errors(tmp_path):
    from ouroboros.task_results import STATUS_COMPLETED

    env = SimpleNamespace(drive_root=tmp_path)

    pipeline._store_task_result(
        env=env,
        task={"id": "task-tool-failed", "type": "task", "text": "make file"},
        text="Created the file.",
        usage={"rounds": 2, "cost": 0.0},
        llm_trace={
            "tool_calls": [{
                "tool": "run_command",
                "args": {"cmd": "python3 -c ..."},
                "result": "⚠️ ARTIFACT_OUTPUT_ERROR: command succeeded but declared output registration failed.",
                "is_error": True,
                "status": "artifact_output_error",
            }],
            "reasoning_notes": [],
        },
        review_evidence={},
    )

    payload = read_result(tmp_path, "task-tool-failed")
    assert payload["status"] == STATUS_COMPLETED
    assert payload["outcome_axes"]["execution"]["status"] == "ok"
    assert payload["outcome_axes"]["objective"]["status"] == "not_evaluated"
    # A command whose declared output never registered stays loud (owner 09.11, A): a promised file is missing.
    assert "details_only" not in payload["outcome_axes"]["execution"]["unresolved_tool_errors"][0]
    assert payload["outcome_axes"]["objective"]["warning"] == "residual_tool_errors_without_review"
    assert payload["reason_code"] == "final_message"
    assert payload["loop_outcome"]["failure"] is None
    assert payload["outcome_axes"]["execution"]["unresolved_tool_errors"][0]["status"] == "artifact_output_error"


def test_store_task_result_allows_recovered_tool_failure_success(tmp_path):
    from ouroboros.task_results import STATUS_COMPLETED

    env = SimpleNamespace(drive_root=tmp_path)

    pipeline._store_task_result(
        env=env,
        task={"id": "task-tool-recovered", "type": "task", "text": "make file"},
        text="Created the file.",
        usage={"rounds": 3, "cost": 0.0},
        llm_trace={
            "tool_calls": [
                {
                    "tool": "edit_text",
                    "args": {"path": "Desktop/report.html"},
                    "result": "⚠️ EDIT_TEXT_ERROR: old_str matched 0 times",
                    "is_error": True,
                    "status": "edit_text_blocked",
                },
                {
                    "tool": "write_file",
                    "args": {"root": "user_files", "path": "Desktop/report.html"},
                    "result": "OK: wrote user_files:Desktop/report.html\nARTIFACT_OUTPUTS: registered user file -> artifact_store:report.html",
                    "is_error": False,
                    "status": "ok",
                    "artifact_registered": True,
                },
            ],
            "reasoning_notes": [],
        },
        review_evidence={},
    )

    payload = read_result(tmp_path, "task-tool-recovered")
    assert payload["status"] == STATUS_COMPLETED
    assert payload["outcome_axes"]["execution"]["status"] == "ok"
    assert payload["outcome_axes"]["objective"]["status"] == "not_evaluated"
    assert payload["loop_outcome"]["failure"] is None


def test_store_task_result_keeps_trace_refs_in_one_place(tmp_path):
    """The stored row names its trace refs once, at the top level; its ``loop_outcome`` carries none.

    The refs grow by one row per model answer and per tool call and the body is rewritten whole on
    every write, so a second copy doubles each rewrite. The caller's outcome object keeps its
    refs, and the top-level reader (a retained call's source) answers alike over this row and
    over a row that holds both copies, which is read as stored.
    """
    from ouroboros.observability import persist_call
    from ouroboros.outcomes import collect_trace_refs, derive_loop_outcome
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.control_task_results import _get_task_result
    from ouroboros.tools.registry import ToolContext

    task_id = "task-trace-refs"
    call = persist_call(tmp_path, task_id=task_id, call_id="tool_read_1", call_type="tool_call",
                        payload={"tool": "read_file", "result": "retained bytes"})
    usage = {"rounds": 1, "cost": 0.0, "execution_id": "exec_trace_refs",
             "llm_call_refs": [{"llm_call_id": "llm_1", "round": 1, "response_ref": {"path": "resp"}}]}
    llm_trace = {"tool_calls": [{"tool": "read_file", "result": "ok", "trace_ref": call}],
                 "reasoning_notes": []}
    outcome = derive_loop_outcome("done", usage, llm_trace)
    refs = collect_trace_refs(usage, llm_trace)
    assert refs["llm_call_refs"] and refs["tool_call_refs"] and outcome["trace_refs"] == refs

    pipeline._store_task_result(
        env=SimpleNamespace(drive_root=tmp_path), task={"id": task_id, "type": "task", "text": "hi"},
        text="done", usage=usage, llm_trace=llm_trace, review_evidence={}, loop_outcome=outcome,
    )

    stored = read_result(tmp_path, task_id)
    assert stored["trace_refs"] == refs
    assert "trace_refs" not in stored["loop_outcome"]
    assert stored["loop_outcome"]["usage"]["total_rounds"] == 1  # the rest of the outcome is stored whole
    assert outcome["trace_refs"] == refs  # the caller's object is not the stored projection

    repo = tmp_path / "repo"
    repo.mkdir()
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path, task_id="parent", budget_drive_root=str(tmp_path),
                      task_metadata={"budget_drive_root": str(tmp_path)})

    def call_source():
        return json.loads(_get_task_result(
            ctx, task_id=task_id, call_id="tool_read_1", manifest_sha256=call["manifest_ref"]["sha256"],
        ))["call_source"]

    single = call_source()
    assert single["task_id"] == task_id and single["call_type"] == "tool_call"
    write_task_result(tmp_path, task_id, stored["status"],
                      loop_outcome={**stored["loop_outcome"], "trace_refs": refs})
    doubled = read_result(tmp_path, task_id)
    assert doubled["trace_refs"] == doubled["loop_outcome"]["trace_refs"] == refs
    assert call_source() == single
