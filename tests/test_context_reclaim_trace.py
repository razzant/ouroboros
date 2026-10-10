"""Focused trace-correlation coverage for reclaim provenance."""

from ouroboros import context_compaction as cc
from ouroboros.loop_tool_execution import process_tool_results


def test_tool_trace_row_carries_exact_tool_call_id_beside_trace_ref():
    trace_ref = {
        "manifest_ref": {"path": "calls/tool.json", "sha256": "a" * 64},
    }
    llm_trace = {"tool_calls": []}
    messages = []

    errors = process_tool_results(
        [{
            "fn_name": "read_file",
            "tool_call_id": "call-correlated",
            "result": "complete actor-visible result",
            "is_error": False,
            "args_for_log": {"path": "README.md"},
            "tool_args": {"path": "README.md"},
            "trace_ref": trace_ref,
            "result_meta": {"status": "ok"},
        }],
        messages,
        llm_trace,
        emit_progress=lambda _message, *, incident=None: None,
    )

    assert errors == 0
    assert messages[0]["tool_call_id"] == "call-correlated"
    row = llm_trace["tool_calls"][0]
    assert row["tool_call_id"] == "call-correlated"
    assert row["trace_ref"] is trace_ref


def test_process_tool_results_accumulates_trace_refs_for_reclaim(tmp_path):
    """The result row owns its invocation-bound trace; the context map stays a diagnostic."""
    from types import SimpleNamespace

    from ouroboros.loop_tool_execution import reclaim_trace_refs

    trace_ref = {"manifest_ref": {"path": "calls/tool.json", "sha256": "b" * 64}}
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tool_result_record import read_tool_result_record
    tools = SimpleNamespace(_ctx=ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="trace-task"))
    messages = []

    errors = process_tool_results(
        [{
            "fn_name": "read_file",
            "tool_call_id": "call-retained", "invocation_id": "invocation-retained",
            "result": "result body",
            "is_error": False,
            "args_for_log": {"path": "README.md"},
            "tool_args": {"path": "README.md"},
            "trace_ref": trace_ref,
        }],
        messages,
        {"tool_calls": []},
        emit_progress=lambda _message, *, incident=None: None,
        tools=tools,
    )

    assert errors == 0
    assert reclaim_trace_refs(tools._ctx) == {"call-retained": trace_ref}
    record = read_tool_result_record(messages[0])
    assert record["state"] == "recorded" and record["trace_ref"] == trace_ref
    assert record["invocation"]["invocation_id"] == "invocation-retained"
    # No tools context: still no crash, accessor stays empty.
    assert reclaim_trace_refs(SimpleNamespace()) == {}


def test_prune_reclaim_trace_refs_drops_ids_absent_from_transcript():
    """After a successful reclaim apply, refs whose tool_call_id left the
    transcript are pruned so the mapping stays bounded by live messages
    instead of growing for the task lifetime (S1 N-2)."""
    from types import SimpleNamespace

    from ouroboros.loop_tool_execution import prune_reclaim_trace_refs, reclaim_trace_refs

    ctx = SimpleNamespace(_tool_trace_refs={
        f"call-{i}": {"manifest_ref": {"path": f"calls/{i}.json", "sha256": "c" * 64}}
        for i in range(50)
    })
    kept = dict(ctx._tool_trace_refs["call-7"])
    messages = [
        {"role": "tool", "tool_call_id": "call-7", "content": "still live"},
        {"role": "assistant", "content": "no tool id"},
        "not-a-dict",
    ]

    prune_reclaim_trace_refs(ctx, messages)

    assert reclaim_trace_refs(ctx) == {"call-7": kept}
    # No refs attribute at all: a no-op, never a crash.
    prune_reclaim_trace_refs(SimpleNamespace(), messages)


def test_materializer_resolves_only_this_invocations_bound_trace_refs():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call-a", "function": {"name": "a", "arguments": "x" * 1_000}},
                {"id": "call-b", "function": {"name": "b", "arguments": "y" * 1_000}},
            ],
        },
        {"role": "tool", "tool_call_id": "call-a", "content": "a" * 1_000},
        {"role": "tool", "tool_call_id": "call-b", "content": "b" * 1_000},
    ]
    ref_a = {"path": "calls/a.json", "sha256": "a" * 64}
    ref_b = {"path": "calls/b.json", "sha256": "b" * 64}
    unrelated = {"path": "calls/other.json", "sha256": "f" * 64}

    # An old call-ID map cannot prove which occurrence supplied these results.
    assert cc._atomic_units(messages, trace_refs_by_tool_call_id={"call-a": ref_a, "call-b": ref_b})[0].source_refs == ()
    from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY, make_tool_result_record
    for index, ref in ((1, ref_a), (2, ref_b)):
        row = messages[index]
        row[TOOL_RESULT_RECORD_KEY] = make_tool_result_record(
            {"tool_call_id": row["tool_call_id"], "invocation_id": f"invoke-{index}", "trace_ref": ref}, row["content"])
    unit = cc._atomic_units(
        messages,
        trace_refs_by_tool_call_id={
            "call-a": unrelated,
            "call-b": ref_b,
            "call-other": unrelated,
        },
    )[0]

    assert unit.source_refs == (ref_a, ref_b)
    assert unrelated not in unit.source_refs
