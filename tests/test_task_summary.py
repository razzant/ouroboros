"""The task-summary synthesis of ``ouroboros.agent_task_pipeline``.

Split out of ``tests/test_agent_task_pipeline.py`` when that module was divided
by theme; every moved block is verbatim. Covers `_run_task_summary` model
routing and its chat-row payload (chat_id, flat snapshot cost fields, outcome
axes), the trivial-task LLM bypass, the multi-round zero-tool prompt, the
review-evidence prompt section and `build_trace_summary` failure facts.
"""

import json

import ouroboros.agent_task_pipeline as pipeline


def test_task_summary_prefers_direct_model_when_openrouter_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "openai::gpt-5.5-mini")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "openai::gpt-5.5-mini")
    monkeypatch.setenv("OUROBOROS_MODEL", "openai::gpt-5.5")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "openai::gpt-5.5")

    captured = {}

    class FakeLlm:
        def chat(self, *, messages, model, reasoning_effort, max_tokens, use_local):
            captured["messages"] = messages
            captured["model"] = model
            captured["reasoning_effort"] = reasoning_effort
            captured["max_tokens"] = max_tokens
            captured["use_local"] = use_local
            return {"content": "direct summary ok"}, {"cost": 0}

    drive_logs = tmp_path / "logs"
    drive_logs.mkdir(parents=True)

    # Use rounds > 1 so the task is non-trivial and the LLM summary path is taken
    pipeline._run_task_summary(
        env=None,
        llm=FakeLlm(),
        task={"id": "task-123", "type": "task", "text": "Reply with exactly OK."},
        usage={"rounds": 3, "cost": 0.01, "result_status": "failed", "reason_code": "empty_final_text"},
        llm_trace={"tool_calls": [{"tool": "read_file", "args": {}}], "reasoning_notes": []},
        drive_logs=drive_logs,
    )

    assert captured["model"] == "openai::gpt-5.5-mini"
    assert captured["use_local"] is False
    chat_lines = (drive_logs / "chat.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(chat_lines) == 1
    payload = json.loads(chat_lines[0])
    assert payload["type"] == "task_summary"
    assert payload["text"] == "direct summary ok"
    # Non-trivial task metadata is persisted
    assert payload["tool_calls"] == 1
    assert payload["rounds"] == 3
    assert payload["outcome_axes"]["execution"]["status"] == "failed"
    assert payload["outcome_axes"]["objective"]["status"] == "not_evaluated"
    assert payload["reason_code"] == "empty_final_text"

def test_task_summary_row_carries_chat_id_for_trivial_task(tmp_path):
    """A trivial task (no tools, <=1 round) skips the LLM summary but still
    stamps the project chat_id, so the summary row routes to its project
    thread on history reload instead of defaulting to the main chat."""
    drive_logs = tmp_path / "logs"
    drive_logs.mkdir(parents=True)
    pipeline._run_task_summary(
        env=None,
        llm=None,
        task={"id": "p1", "type": "task", "text": "hi", "chat_id": 1234},
        usage={"rounds": 1, "cost": 0.0},
        llm_trace={"tool_calls": [], "reasoning_notes": []},
        drive_logs=drive_logs,
    )
    rows = [
        json.loads(line)
        for line in (drive_logs / "chat.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summaries = [r for r in rows if r.get("type") == "task_summary"]
    assert summaries and summaries[0]["chat_id"] == 1234

def test_task_summary_row_carries_flat_snapshot_cost_fields(tmp_path):
    """v6.82 P1: the task_summary chat row carries the pre-synthesis snapshot's
    flat cost fields (previously discarded into prose) so history replay can
    show honest card cost. Fields absent from the snapshot (cost_usd,
    cost_accounting_error) are never fabricated."""
    drive_logs = tmp_path / "logs"
    drive_logs.mkdir(parents=True)
    snapshot_usage = {
        "rounds": 1,
        "cost": 0.0,
        # _pre_synthesis_usage_snapshot root-shape keys:
        "cost_snapshot_at": "2026-07-29T00:00:00Z",
        "cost_final": False,
        "cost_with_children_partial": True,
        "accounted_upper_bound_usd_with_children": 1.25,
        "reserved_usd": 0.1,
        "unresolved_upper_bound_usd": 0.2,
        "unknown_unmetered": 0,
        "ledger_integrity": "ok",
        "cost_accounting_status": "available",
    }
    pipeline._run_task_summary(
        env=None,
        llm=None,
        task={"id": "p2", "type": "task", "text": "hi", "chat_id": 1},
        usage=snapshot_usage,
        llm_trace={"tool_calls": [], "reasoning_notes": []},
        drive_logs=drive_logs,
    )
    rows = [
        json.loads(line)
        for line in (drive_logs / "chat.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    row = next(r for r in rows if r.get("type") == "task_summary")
    assert row["cost_final"] is False
    assert row["cost_with_children_partial"] is True
    # ABI-3 fix-round-2: the snapshot producer emits the honest name only
    # (the legacy fixture spelling here was stale).
    assert row["accounted_upper_bound_usd_with_children"] == 1.25
    assert "cost_usd_with_children" not in row
    assert row["reserved_usd"] == 0.1
    assert row["unresolved_upper_bound_usd"] == 0.2
    assert row["unknown_unmetered"] == 0
    assert row["cost_accounting_status"] == "available"
    assert "cost_usd" not in row
    assert "cost_accounting_error" not in row

def test_task_summary_uses_configured_light_model_when_openrouter_present(monkeypatch):
    from ouroboros.consolidator import _consolidation_route

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    # Unprefixed provider/model ids use OpenRouter, so this Light model is
    # credentialed by the key above and MUST be kept verbatim. An ``openai::``
    # id would select the direct OpenAI transport instead — uncredentialed here
    # (no OPENAI_API_KEY) — and the documented provider-independence fallback in
    # resolve_credentialed_model() would then rewrite it to the first credentialed
    # slot, making the assertion depend on ambient OUROBOROS_MODEL* env leaked by
    # earlier tests in the same worker (the chronic v6.64.2..v6.65.4 CI red).
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "openai/gpt-5.5-mini")

    assert _consolidation_route() == ("openai/gpt-5.5-mini", False)

def test_task_summary_accepts_openai_compatible_when_legacy_base_url_is_present(monkeypatch):
    from ouroboros.consolidator import _consolidation_route

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "anthropic/claude-opus-4.6")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "openai-compatible::custom-model")
    monkeypatch.setenv("OUROBOROS_MODEL", "anthropic/claude-opus-4.6")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "anthropic/claude-opus-4.6")

    assert _consolidation_route() == ("openai-compatible::custom-model", False)

def test_build_trace_summary_shows_structured_failure_facts():
    trace = {
        "tool_calls": [{
            "tool": "run_command",
            "args": {"cmd": ["npm", "install", "-g", "@anthropic-ai/claude-code"]},
            "result": "⚠️ SHELL_EXIT_ERROR: command exited with exit_code=-9 (signal=SIGKILL).",
            "is_error": True,
            "status": "non_zero_exit",
            "exit_code": -9,
            "signal": "SIGKILL",
        }],
        "reasoning_notes": ["Thought this might still work."],
    }

    summary = pipeline.build_trace_summary(trace)

    assert "status=non_zero_exit" in summary
    assert "exit_code=-9" in summary
    assert "signal=SIGKILL" in summary
    assert "Agent notes (supplementary, not source of truth)" in summary

    long_trace = {
        "tool_calls": [
            {
                "tool": "run_command",
                "args": {"cmd": "x" * 5000},
                "is_error": False,
            }
            for _ in range(40)
        ],
        "reasoning_notes": ["note" * 2000],
    }
    assert "OMISSION NOTE" in pipeline.build_trace_summary(long_trace)

def test_task_summary_prompt_includes_review_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "openai::gpt-5.5-mini")

    captured = {}

    class FakeLlm:
        def chat(self, *, messages, model, reasoning_effort, max_tokens, use_local):
            captured["prompt"] = messages[0]["content"]
            return {"content": "summary with review evidence"}, {"cost": 0}

    drive_logs = tmp_path / "logs"
    drive_logs.mkdir(parents=True)

    pipeline._run_task_summary(
        env=None,
        llm=FakeLlm(),
        task={"id": "task-review", "type": "task", "text": "Fix commit flow"},
        usage={"rounds": 4, "cost": 0.02},
        llm_trace={"tool_calls": [{"tool": "commit_reviewed", "args": {}}], "reasoning_notes": []},
        drive_logs=drive_logs,
        review_evidence={
            "has_evidence": True,
            "recent_attempts": [{
                "status": "blocked",
                "critical_findings": [{
                    "severity": "critical",
                    "item": "tests_affected",
                    "reason": "broken",
                }],
            }],
        },
    )

    assert "Structured review evidence" in captured["prompt"]
    assert "tests_affected" in captured["prompt"]
    assert "critical" in captured["prompt"]
    assert "meta-reflection" in captured["prompt"].lower()
    assert "What friction, errors, or weak assumptions slowed the work?" in captured["prompt"]
    assert "What should Ouroboros change in its own process or prompts" in captured["prompt"]
    assert "keep it to 1-2 sentences and DO NOT add meta-reflection" in captured["prompt"]

def test_trivial_task_summary_bypasses_llm_and_uses_short_format(tmp_path):
    class FailIfCalledLlm:
        def chat(self, *args, **kwargs):  # pragma: no cover - should never be called
            raise AssertionError("LLM summary path must be skipped for trivial tasks")

    drive_logs = tmp_path / "logs"
    drive_logs.mkdir(parents=True)

    pipeline._run_task_summary(
        env=None,
        llm=FailIfCalledLlm(),
        task={"id": "task-trivial", "type": "task", "text": "Say hi"},
        usage={"rounds": 1, "cost": 0.0, "result_status": "infra_failed", "reason_code": "llm_api_error"},
        llm_trace={"tool_calls": [], "reasoning_notes": []},
        drive_logs=drive_logs,
    )

    payload = json.loads((drive_logs / "chat.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert payload["type"] == "task_summary"
    assert payload["task_id"] == "task-trivial"
    assert payload["text"] == "Task task-trivial (task): Say hi. 1r, $0.00."
    assert payload["tool_calls"] == 0
    assert payload["rounds"] == 1
    assert payload["outcome_axes"]["execution"]["status"] == "infra_failed"
    assert payload["outcome_axes"]["objective"]["status"] == "not_evaluated"
    assert payload["reason_code"] == "llm_api_error"

def test_multi_round_zero_tool_task_uses_llm_summary_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "openai::gpt-5.5-mini")

    captured = {}

    class FakeLlm:
        def chat(self, *, messages, model, reasoning_effort, max_tokens, use_local):
            captured["prompt"] = messages[0]["content"]
            return {"content": "multi-round summary"}, {"cost": 0}

    drive_logs = tmp_path / "logs"
    drive_logs.mkdir(parents=True)

    pipeline._run_task_summary(
        env=None,
        llm=FakeLlm(),
        task={"id": "task-zero-tool-multi-round", "type": "task", "text": "Think carefully"},
        usage={"rounds": 3, "cost": 0.01},
        llm_trace={"tool_calls": [], "reasoning_notes": ["note"]},
        drive_logs=drive_logs,
    )

    assert "0 tool calls and ≤1 round" in captured["prompt"]
    assert "DO NOT add meta-reflection" in captured["prompt"]
    payload = json.loads((drive_logs / "chat.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert payload["text"] == "multi-round summary"
    assert payload["tool_calls"] == 0
    assert payload["rounds"] == 3
