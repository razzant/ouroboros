"""Production-shaped source-handle regressions for continuity Phase 3B."""

from __future__ import annotations

import json
from types import SimpleNamespace

from ouroboros import context_compaction as compaction
from ouroboros.artifacts import collect_task_artifact_records
from ouroboros.context_budget import ContextReclaimRequest
from ouroboros.consolidator import consolidate_scratchpad
from ouroboros.loop_tool_execution import _truncate_tool_result, process_tool_results
from ouroboros.memory import Memory
from ouroboros.review_evidence import build_task_acceptance_evidence
from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request
from ouroboros.tools.core import _read_file
from ouroboros.tools.registry import ToolContext


def _tool_ctx(tmp_path, *, task_id: str = "source-handles") -> ToolContext:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return ToolContext(repo_dir=repo, drive_root=tmp_path, task_id=task_id)


def _source_ref_from_visible_result(text: str) -> dict:
    prefix = "FULL_RESULT_SOURCE_JSON="
    line = next(line for line in text.splitlines() if line.startswith(prefix))
    return json.loads(line[len(prefix):])


def _read_source(ctx: ToolContext, ref: dict, *, start_char: int = 0) -> str:
    read = ref["read"]
    assert read["tool"] == "read_file"
    args = dict(read["arguments"])
    args["start_char"] = start_char
    return _read_file(ctx, **args)


def test_fifo_eviction_keeps_exact_block_and_current_scratchpad_names_reader(tmp_path):
    memory = Memory(tmp_path)
    contents = [f"scratch-source-{index}" for index in range(11)]
    for content in contents:
        memory.append_scratchpad_block(content, source="phase3b-test")

    current = memory.load_scratchpad()
    assert "read_file(root='runtime_data', path='memory/scratchpad_journal.jsonl'" in current

    journal = _read_file(
        _tool_ctx(tmp_path),
        root="runtime_data",
        path="memory/scratchpad_journal.jsonl",
    )
    rows = [json.loads(line) for line in journal.splitlines()[1:] if line.startswith("{")]
    evicted = [row for row in rows if row.get("type") == "block_evicted"]
    assert [row["evicted_block_content"] for row in evicted] == [contents[0]]


class _ConsolidationLLM:
    def chat(self, **_kwargs):
        return {
            "content": json.dumps({
                "knowledge_entries": [],
                "compressed_block": "compressed working memory",
            })
        }, {"prompt_tokens": 10, "completion_tokens": 5}


def test_scratchpad_consolidation_journals_exact_replaced_blocks_and_ref(tmp_path):
    memory = Memory(tmp_path)
    original = []
    for index in range(4):
        content = f"block-{index}-" + (chr(97 + index) * 8_000)
        original.append(memory.append_scratchpad_block(content, source=f"source-{index}"))

    usage = consolidate_scratchpad(
        memory,
        tmp_path / "memory" / "knowledge",
        _ConsolidationLLM(),
    )
    assert usage == {"prompt_tokens": 10, "completion_tokens": 5}

    blocks = memory.load_scratchpad_blocks()
    consolidated = blocks[0]
    assert consolidated["source"] == "consolidation"
    source_ref = consolidated["metadata"]["source_ref"]
    assert source_ref["read"] == {
        "tool": "read_file",
        "arguments": {
            "root": "runtime_data",
            "path": "memory/scratchpad_journal.jsonl",
            "start_line": 1,
        },
    }
    assert source_ref["entry_id"]

    journal = _read_file(
        _tool_ctx(tmp_path),
        root="runtime_data",
        path="memory/scratchpad_journal.jsonl",
    )
    rows = [json.loads(line) for line in journal.splitlines()[1:] if line.startswith("{")]
    entry = next(row for row in rows if row.get("entry_id") == source_ref["entry_id"])
    assert entry["type"] == "blocks_consolidated"
    assert entry["source_blocks"] == original[:2]
    assert source_ref["entry_id"] in memory.load_scratchpad()


def _project_large_result(tmp_path, *, tool_name: str, call_id: str, result: str):
    ctx = _tool_ctx(tmp_path, task_id="large-result")
    messages: list[dict] = []
    trace = {"tool_calls": []}
    tools = SimpleNamespace(_ctx=ctx)
    process_tool_results(
        [{
            "fn_name": tool_name,
            "tool_call_id": call_id,
            "result": result,
            "is_error": False,
            "tool_args": {"cmd": "non-idempotent-operation"},
            "args_for_log": {"cmd": "non-idempotent-operation"},
            "trace_ref": {"manifest_ref": {"path": "private-only"}},
            "result_meta": {"status": "ok"},
        }],
        messages,
        trace,
        emit_progress=lambda _message, *, incident=None: None,
        tools=tools,
    )
    return ctx, messages[0]["content"], trace["tool_calls"][0]


def test_100k_non_idempotent_command_has_exact_actor_read_handle(tmp_path):
    decisive_suffix = "\nDECISIVE_SUFFIX: transaction committed but verification FAILED"
    full = "command-issued-once\n" + ("x" * 100_000) + decisive_suffix
    ctx, visible, trace_row = _project_large_result(
        tmp_path,
        tool_name="run_command",
        call_id="call-non-idempotent",
        result=full,
    )

    assert decisive_suffix not in visible
    assert "Do not rerun this tool to recover omitted output." in visible
    ref = _source_ref_from_visible_result(visible)
    assert trace_row["result_partial"] is True
    assert trace_row["result_source_ref"] == ref
    assert ref["root"] == "artifact_store"
    assert ref["size"] == len(full.encode("utf-8"))
    assert decisive_suffix in _read_source(ctx, ref, start_char=95_000)


def test_large_extension_result_uses_same_exact_actor_read_handle(tmp_path):
    decisive_suffix = "\nEXTENSION_DECISION=DENY"
    token = "sk-" + ("secret" * 8)
    full = json.dumps({
        "payload": "y" * 20_000, "api_key": token, "decision": decisive_suffix,
    })
    ctx, visible, trace_row = _project_large_result(
        tmp_path,
        tool_name="ext_demo_large_result",
        call_id="call-extension",
        result=full,
    )

    assert decisive_suffix not in visible
    ref = _source_ref_from_visible_result(visible)
    assert trace_row["result_partial"] is True
    assert trace_row["result_source_ref"] == ref
    recovered = _read_source(ctx, ref)
    assert full in recovered
    evidence = build_task_acceptance_evidence(
        ctx, llm_trace={"tool_calls": [trace_row]},
        drive_root=tmp_path, task_id="large-result",
    )
    assert token not in json.dumps(evidence, ensure_ascii=False)
    assert "***REDACTED***" in evidence["tool_trajectory"][0]["result"]
    assert collect_task_artifact_records(tmp_path, "large-result") == []


def _unit(call_id: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": f"reasoning-{call_id}",
            "tool_calls": [{
                "id": call_id,
                "function": {
                    "name": "run_command",
                    "arguments": json.dumps({"cmd": "one-shot"}),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": "z" * 6_000 + f"-exact-result-tail-{call_id}",
        },
    ]


def test_context_capsule_checkpoint_is_actor_readable_and_dangling_is_explicit(
    monkeypatch, tmp_path,
):
    messages = _unit("checkpointed")
    request = ContextReclaimRequest(
        route_fp="main-route",
        round_id="round-1",
        transcript_sha256=compaction.context_reclaim_transcript_sha256(messages),
        measurement_basis="cold_estimate",
        measurement_density=1.0,
        reclaim_goal_tokens=100,
        allow_partial_shrink=True,
    )
    monkeypatch.setattr(compaction, "_summarizer_spec", lambda: {
        "model": "summary-model",
        "resolved_model": "summary-model",
        "provider": "test",
        "route_fp": "summary-route",
        "effort": "low",
        "output_budget": 32_768,
        "use_local": False,
    })
    monkeypatch.setattr(
        compaction,
        "_call_summarizer",
        lambda parts, **_kwargs: {
            part.source_id: f"summary {part.sha256}" for part in parts
        },
    )

    rebuilt, receipt, _usage = compaction.compact_tool_history_llm(
        messages,
        request=request,
        drive_root=tmp_path,
        task_id="checkpoint-task",
        keep_recent=0,
        negative_memo=set(),
    )
    assert receipt.status == "applied"
    capsule = rebuilt[0]["content"][0]["_context_capsule"]
    ref = capsule["checkpoint_ref"]
    assert ref["root"] == "artifact_store"
    ctx = _tool_ctx(tmp_path, task_id="checkpoint-task")
    recovered = _read_source(ctx, ref)
    checkpoint = json.loads(recovered.split("\n", 1)[1])
    assert checkpoint["messages"] == messages

    source_path = tmp_path / "task_results" / "artifacts" / "checkpoint-task" / ref["path"]
    source_path.unlink()
    assert "NOT_FOUND" in _read_source(ctx, ref)


class _MustNotReviewPartial:
    def __init__(self):
        self.calls = 0

    def chat(self, **_kwargs):
        self.calls += 1
        raise AssertionError("an unresolved partial source must not reach a clean reviewer")


def test_metadata_less_actor_truncation_envelope_abstains_before_review(tmp_path):
    actor_view = _truncate_tool_result("legacy" * 100_000, "run_command")
    assert "truncated from 600000" in actor_view
    ctx = _tool_ctx(tmp_path, task_id="legacy-partial")
    evidence = build_task_acceptance_evidence(
        ctx,
        llm_trace={"tool_calls": [{
            "tool": "run_command", "status": "ok", "result": actor_view,
        }]},
        drive_root=tmp_path,
        task_id="legacy-partial",
    )
    row = evidence["tool_trajectory"][0]
    assert "truncated from 600000" in row["result"]
    assert row["result_complete"] is False
    assert evidence["__unresolved_partial_artifacts__"][0]["status"] == "source_unavailable"

    llm = _MustNotReviewPartial()
    result = run_review_request(
        ReviewRequest(
            surface="task_acceptance", goal="decide from legacy evidence",
            subject="candidate", evidence=evidence,
            policy={"min_successful_slots": 1}, task_id="legacy-partial",
        ),
        slots=[ReviewSlot(slot_id="slot", model="review-model")],
        drive_root=tmp_path,
        llm=llm,
    )
    assert llm.calls == 0
    assert result.aggregate_signal == "DEGRADED"
    assert result.degraded is True


def test_explicit_complete_row_is_not_reclassified_from_envelope_text(tmp_path):
    result_text = "0123456789\n... (truncated from 20 chars, limit=10)"
    ctx = _tool_ctx(tmp_path, task_id="explicit-complete")
    evidence = build_task_acceptance_evidence(
        ctx,
        llm_trace={"tool_calls": [{
            "tool": "run_command", "status": "ok", "result": result_text,
            "result_partial": False,
        }]},
        drive_root=tmp_path,
        task_id="explicit-complete",
    )
    assert "__unresolved_partial_artifacts__" not in evidence
    assert "result_complete" not in evidence["tool_trajectory"][0]


def test_budget_recap_preserves_every_legacy_actor_envelope(tmp_path):
    calls = []
    for index in range(3):
        actor_view = _truncate_tool_result(str(index) * 500_000, "run_command")
        assert "truncated from 500000" in actor_view
        calls.append({"tool": "run_command", "status": "ok", "result": actor_view})

    evidence = build_task_acceptance_evidence(
        _tool_ctx(tmp_path, task_id="legacy-budget"),
        llm_trace={"tool_calls": calls},
        drive_root=tmp_path,
        task_id="legacy-budget",
    )
    rows = evidence["tool_trajectory"]
    assert len(rows) == 3
    assert all("truncated from 500000" in row["result"] for row in rows)
    assert all(row["result_complete"] is False for row in rows)
    assert all("_legacy_projection_envelope" not in row for row in rows)
    assert len(evidence["__unresolved_partial_artifacts__"]) >= 3


def test_redaction_expansion_preserves_legacy_actor_envelope(tmp_path):
    credential_url = "https://alice:phase3b-secret@example.invalid/private?"
    full = (credential_url * 3_000) + ("Z" * 400_000)
    actor_view = _truncate_tool_result(full, "run_command")
    marker = f"truncated from {len(full)}"
    assert marker in actor_view

    evidence = build_task_acceptance_evidence(
        _tool_ctx(tmp_path, task_id="legacy-redaction"),
        llm_trace={"tool_calls": [{
            "tool": "run_command", "status": "ok", "result": actor_view,
        }]},
        drive_root=tmp_path,
        task_id="legacy-redaction",
    )
    row = evidence["tool_trajectory"][0]
    assert "phase3b-secret" not in row["result"]
    assert "***REDACTED***" in row["result"]
    assert marker in row["result"]
    assert row["result_complete"] is False
    assert "_legacy_projection_envelope" not in row


def test_task_acceptance_abstains_before_review_on_unresolved_partial_source(tmp_path):
    full = "decision-input\n" + ("p" * 20_000) + "\nDECISIVE_ACCEPTANCE_SUFFIX=FAIL"
    ctx, _visible, trace_row = _project_large_result(
        tmp_path, tool_name="ext_acceptance_probe", call_id="acceptance-source", result=full,
    )
    trace = {"tool_calls": [trace_row]}
    complete_evidence = build_task_acceptance_evidence(
        ctx, llm_trace=trace, drive_root=tmp_path, task_id="large-result",
    )
    assert complete_evidence["tool_trajectory"][0]["result_complete"] is True
    assert "DECISIVE_ACCEPTANCE_SUFFIX=FAIL" in complete_evidence["tool_trajectory"][0]["result"]

    source_ref = trace_row["result_source_ref"]
    source_path = tmp_path / "task_results" / "artifacts" / "large-result" / source_ref["path"]
    source_path.unlink()
    evidence = build_task_acceptance_evidence(
        ctx, llm_trace=trace, drive_root=tmp_path, task_id="large-result",
    )
    assert evidence["__unresolved_partial_artifacts__"][0]["status"] == "source_unavailable"

    llm = _MustNotReviewPartial()
    paid = []
    result = run_review_request(
        ReviewRequest(
            surface="task_acceptance",
            goal="decide from evidence",
            subject="candidate",
            evidence=evidence,
            policy={"min_successful_slots": 1},
            task_id="acceptance-partial",
        ),
        slots=[ReviewSlot(slot_id="slot", model="review-model")],
        drive_root=tmp_path,
        llm=llm,
        usage_ctx=SimpleNamespace(_review_paid_stamp=lambda: paid.append(True)),
    )

    assert llm.calls == 0
    assert paid == []
    assert result.aggregate_signal == "DEGRADED"
    assert result.degraded is True
    assert result.actors[0]["status"] == "not_dispatched"
    assert result.actors[0]["signal"] == "DEGRADED"


def test_marker_only_repo_diff_abstains_before_review(monkeypatch, tmp_path):
    """A legacy diff producer must not turn its visible slice into PASS."""
    import ouroboros.review_evidence as evidence_mod

    monkeypatch.setattr(
        evidence_mod,
        "collect_turn_diff",
        lambda _ctx, **_kwargs: (
            "diff --git a/visible.py b/visible.py\n"
            "⚠️ OMISSION NOTE: truncated at 20000 chars; original length 80000"
        ),
    )
    ctx = _tool_ctx(tmp_path, task_id="marker-only-diff")
    evidence = build_task_acceptance_evidence(
        ctx, llm_trace={"tool_calls": []}, drive_root=tmp_path, task_id="marker-only-diff",
    )
    assert evidence["__unresolved_partial_artifacts__"][0]["tool"] == "repo_diff"

    llm = _MustNotReviewPartial()
    result = run_review_request(
        ReviewRequest(
            surface="task_acceptance", goal="decide from evidence", subject="candidate",
            evidence=evidence, policy={"min_successful_slots": 1}, task_id="marker-only-diff",
        ),
        slots=[ReviewSlot(slot_id="slot", model="review-model")],
        drive_root=tmp_path,
        llm=llm,
    )
    assert llm.calls == 0
    assert result.aggregate_signal == "DEGRADED"


def test_large_repo_diff_materializes_exact_source_and_keeps_pass_path(tmp_path):
    """The normal over-limit path is exact-source backed, not silently degraded."""
    import subprocess as sp

    ctx = _tool_ctx(tmp_path, task_id="exact-repo-diff")
    repo = ctx.repo_dir
    sp.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "large.py").write_text("x = 0\n", encoding="utf-8")
    sp.run(["git", "add", "large.py"], cwd=repo, check=True, capture_output=True)
    sp.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "base"],
        cwd=repo, check=True, capture_output=True,
    )
    (repo / "large.py").write_text(
        "\n".join(f"value_{i} = {i}" for i in range(5000)) + "\nDECISIVE_DIFF_TAIL=present\n",
        encoding="utf-8",
    )
    evidence = build_task_acceptance_evidence(
        ctx, llm_trace={"tool_calls": []}, drive_root=tmp_path, task_id="exact-repo-diff",
    )
    assert "__unresolved_partial_artifacts__" not in evidence
    assert evidence["repo_diff_source_ref"]["kind"] == "task_source"
    assert "DECISIVE_DIFF_TAIL=present" in evidence["repo_diff"]
    assert collect_task_artifact_records(tmp_path, "exact-repo-diff") == []

    class _PassLLM:
        def chat(self, **_kwargs):
            return {"content": json.dumps({"verdict": "PASS", "findings": [], "summary": "ok"})}, {}

    result = run_review_request(
        ReviewRequest(
            surface="task_acceptance", goal="decide from evidence", subject="candidate",
            evidence=evidence, policy={"min_successful_slots": 1}, task_id="exact-repo-diff",
        ),
        slots=[ReviewSlot(slot_id="slot", model="review-model")],
        drive_root=tmp_path,
        llm=_PassLLM(),
    )
    assert result.aggregate_signal == "PASS"


# ── AP3: a pageable tool result persists its exact source like any other ──────

class _CountingLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, **_kwargs):
        self.calls += 1
        body = {"verdict": "PASS", "findings": [], "summary": "reviewed"}
        return {"content": json.dumps(body)}, {"prompt_tokens": 10, "completion_tokens": 5}


def _dispatch_acceptance(tmp_path, evidence: dict, llm) -> object:
    return run_review_request(
        ReviewRequest(
            surface="task_acceptance", goal="decide", subject="candidate",
            evidence=evidence, policy={"min_successful_slots": 1}, task_id="pageable",
        ),
        slots=[ReviewSlot(slot_id="slot", model="review-model")],
        drive_root=tmp_path,
        llm=llm,
    )


def test_over_limit_pageable_results_persist_their_exact_source(tmp_path):
    """The agent can page `read_file`; the acceptance DECIDER cannot. Skipping
    persistence for pageable tools left the panel with a partial row whose exact
    source did not exist, which refused the whole panel for nothing."""
    for tool_name in ("read_file", "query_code", "chat_history"):
        ctx, visible, trace_row = _project_large_result(
            tmp_path, tool_name=tool_name, call_id=f"page-{tool_name}",
            result="page-head\n" + ("y" * 120_000) + "\nDECISIVE_TAIL=FAIL",
        )
        assert trace_row["result_partial"] is True
        assert trace_row["result_source_status"] == "ready"
        assert "FULL_RESULT_SOURCE_JSON=" in visible
        assert "exact source persistence failed" not in visible
        # The wording keeps the tool's own affordance instead of forbidding it.
        assert "or page this tool (offset/limit) for the omitted range" in visible
        ref = _source_ref_from_visible_result(visible)
        assert "DECISIVE_TAIL=FAIL" in _read_source(ctx, ref, start_char=100_000)

        evidence = build_task_acceptance_evidence(
            ctx, llm_trace={"tool_calls": [dict(trace_row)]},
            drive_root=tmp_path, task_id=ctx.task_id,
        )
        assert evidence["tool_trajectory"][0]["result_complete"] is True
        assert "__unresolved_partial_artifacts__" not in evidence

        llm = _CountingLLM()
        result = _dispatch_acceptance(tmp_path, evidence, llm)
        assert llm.calls == 1
        assert result.aggregate_signal == "PASS"


def test_a_non_pageable_tool_keeps_its_do_not_rerun_wording(tmp_path):
    _ctx, visible, _row = _project_large_result(
        tmp_path, tool_name="run_command", call_id="page-run",
        result="cmd\n" + ("z" * 120_000),
    )
    assert "Do not rerun this tool to recover omitted output." in visible
    assert "or page this tool" not in visible


def test_a_budget_shed_row_dispatches_while_a_missing_source_still_refuses(tmp_path):
    shed_only = {
        "task_contract": {"requirements": "do X"},
        "__provenance__": {},
        "__unresolved_partial_artifacts__": [{
            "tool": "read_file", "status": "not_materialized_for_reviewer",
            "source_ref": {"kind": "artifact", "path": "tool_results/page.txt"},
        }],
    }
    llm = _CountingLLM()
    assert _dispatch_acceptance(tmp_path, shed_only, llm).aggregate_signal == "PASS"
    assert llm.calls == 1

    genuine = {
        **shed_only,
        "__unresolved_partial_artifacts__": [
            {"tool": "read_file", "status": "source_unavailable", "source_ref": {}},
        ],
    }
    refusing = _MustNotReviewPartial()
    result = _dispatch_acceptance(tmp_path, genuine, refusing)
    assert refusing.calls == 0
    assert result.aggregate_signal == "DEGRADED"
