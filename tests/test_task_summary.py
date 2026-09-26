"""The free host facts row of ``ouroboros.agent_task_pipeline`` (no paid narrative).

Owner decision 2=A (TZ-2 C5) removed the paid "task summary" narrative. What
survives is `_record_task_facts`: one ``task_summary`` chat row of kind
``host_task_facts`` with the facts its readers need (chat_id, flat snapshot cost
fields, outcome axes, tool metrics, routing) and no prose, never labelled as an
authored narrative. Also covers the Light consolidation route the remaining
chat consolidation uses and `build_trace_summary` failure facts.
"""

import json
from types import SimpleNamespace

import pytest

import ouroboros.agent_task_pipeline as pipeline


def _rows(drive_logs):
    return [json.loads(line) for line in (drive_logs / "chat.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def no_model_calls(monkeypatch):
    monkeypatch.setattr("ouroboros.llm_observability.chat_observed",
                        lambda *_a, **_k: pytest.fail("the facts row buys no model call"))
    monkeypatch.setattr("ouroboros.llm.LLMClient", lambda *_a, **_k: pytest.fail("the facts row needs no client"))


def test_facts_row_buys_no_model_call_and_carries_the_facts(tmp_path, no_model_calls):
    drive_logs = tmp_path / "logs"
    drive_logs.mkdir(parents=True)

    # Non-trivial (several rounds, a tool call): the old paid narrative path.
    pipeline._record_task_facts(
        env=None,
        task={"id": "task-123", "type": "task", "text": "Reply with exactly OK."},
        usage={"rounds": 3, "cost": 0.01, "result_status": "failed", "reason_code": "empty_final_text"},
        llm_trace={"tool_calls": [{"tool": "read_file", "args": {}}], "reasoning_notes": []},
        drive_logs=drive_logs,
    )

    [payload] = _rows(drive_logs)
    assert payload["type"] == "task_summary"
    assert payload["summary_kind"] == "host_task_facts"
    assert payload["summary_id"] == "task-facts:task-123"
    assert payload["text"] == ""  # no prose: the task text is not retold either
    assert payload["outcome_final"] is False
    assert payload["tool_calls"] == 1 and payload["tool_call_counts"] == {"read_file": 1}
    assert payload["rounds"] == 3
    assert payload["outcome_axes"]["execution"]["status"] == "failed"
    assert payload["outcome_axes"]["objective"]["status"] == "not_evaluated"
    assert payload["reason_code"] == "empty_final_text"
    assert "source_coverage" not in payload


def test_facts_row_is_never_an_authored_narrative_while_legacy_rows_still_resolve(tmp_path, no_model_calls):
    from ouroboros.main_context_authority import project_main_task_authority
    from ouroboros.project_dialogue import append_canonical_task_summary
    from ouroboros.task_results import load_task_result, write_task_result

    ref = {"kind": "task_result", "task_id": "", "reader": "get_task_result"}
    for task_id in ("new-root", "legacy-root"):
        write_task_result(tmp_path, task_id, "completed", result="R" * 200001)
    pipeline._record_task_facts(
        SimpleNamespace(drive_root=tmp_path),
        {"id": "new-root", "root_task_id": "new-root", "type": "task", "chat_id": 1, "text": "work"},
        {"rounds": 4, "cost": 0.0}, {"tool_calls": [{"tool": "read_file"}]}, tmp_path / "logs",
    )
    # A historical paid narrative keeps resolving through the unchanged reader.
    legacy_ref = {**ref, "task_id": "legacy-root"}
    assert append_canonical_task_summary(tmp_path, {
        "type": "task_summary", "summary_kind": "authored_root_summary",
        "summary_id": "task-narrative:legacy-root", "task_id": "legacy-root",
        "result_ref": legacy_ref, "source_coverage": {"task_result": legacy_ref},
        "text": "Legacy authored account",
    })

    assert "continuation_narrative" not in load_task_result(tmp_path, "new-root")

    def projected(task_id):
        authority = {"task_id": task_id, "result": "R" * 200001, "task_contract": {"objective": "old"},
                     "source": {**ref, "task_id": task_id, "arguments": {"task_id": task_id, "include_authority": True}}}
        return project_main_task_authority(
            {"id": "next", "predecessor_authority": authority}, drive_root=tmp_path,
        )["predecessor_authority"]["result"]

    new = projected("new-root")
    assert new["narrative_status"] == "unavailable"
    assert new["narrative_gap"]["kind"] == "continuation_narrative_unavailable"
    legacy = projected("legacy-root")
    assert legacy["narrative_status"] == "available"
    assert legacy["narrative"]["text"] == "Legacy authored account"


def test_facts_row_has_no_visible_summary_even_for_a_project(tmp_path, no_model_calls):
    drive_logs = tmp_path / "logs"
    pipeline._record_task_facts(
        None, {"id": "bound", "type": "task", "text": "Ship it", "chat_id": 1, "project_id": "launch"},
        {"rounds": 5, "cost": 0.0}, {"tool_calls": []}, drive_logs,
    )
    pipeline._record_task_facts(
        None, {"id": "unbound", "type": "task", "text": "Ship it", "chat_id": 1},
        {"rounds": 5, "cost": 0.0}, {"tool_calls": []}, drive_logs,
    )
    texts = {row["task_id"]: row["text"] for row in _rows(drive_logs)}
    assert texts == {"bound": "", "unbound": ""}


def test_facts_row_carries_chat_id_for_trivial_task(tmp_path, no_model_calls):
    """The facts row stamps the project chat_id, so it routes to its project
    thread on history reload instead of defaulting to the main chat."""
    drive_logs = tmp_path / "logs"
    drive_logs.mkdir(parents=True)
    pipeline._record_task_facts(
        env=None,
        task={"id": "p1", "type": "task", "text": "hi", "chat_id": 1234},
        usage={"rounds": 1, "cost": 0.0, "result_status": "infra_failed", "reason_code": "llm_api_error"},
        llm_trace={"tool_calls": [], "reasoning_notes": []},
        drive_logs=drive_logs,
    )
    [summary] = [r for r in _rows(drive_logs) if r.get("type") == "task_summary"]
    assert summary["chat_id"] == 1234
    assert summary["text"] == ""  # the former trivial-task host line is gone
    assert summary["tool_calls"] == 0 and summary["rounds"] == 1
    assert summary["outcome_axes"]["execution"]["status"] == "infra_failed"
    assert summary["reason_code"] == "llm_api_error"


def test_facts_row_carries_flat_snapshot_cost_fields(tmp_path):
    """v6.82 P1: the task_summary chat row carries the pre-synthesis snapshot's
    flat cost fields so history replay can show honest card cost. Fields absent
    from the snapshot (cost_usd, cost_accounting_error) are never fabricated."""
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
    pipeline._record_task_facts(
        env=None,
        task={"id": "p2", "type": "task", "text": "hi", "chat_id": 1},
        usage=snapshot_usage,
        llm_trace={"tool_calls": [], "reasoning_notes": []},
        drive_logs=drive_logs,
    )
    row = next(r for r in _rows(drive_logs) if r.get("type") == "task_summary")
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


def test_facts_row_failure_is_contained(tmp_path, monkeypatch, caplog):
    """A failed append names the task and never raises into post-task work."""
    import ouroboros.project_dialogue as dialogue

    monkeypatch.setattr(dialogue, "append_canonical_task_summary",
                        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")))
    with caplog.at_level("WARNING"):
        pipeline._record_task_facts(None, {"id": "full-disk", "chat_id": 1}, {"rounds": 2}, {"tool_calls": []},
                                    tmp_path / "logs")
    assert "full-disk" in caplog.text


def test_consolidation_route_uses_configured_light_model_when_openrouter_present(monkeypatch):
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


def test_consolidation_route_accepts_openai_compatible_when_legacy_base_url_is_present(monkeypatch):
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


def test_facts_row_states_files_rescued_from_a_stat_only_walk(tmp_path, no_model_calls):
    """TZ-2 C2: at terminal the free facts row says how many files reached the task's
    artifact store — a positive count, a confirmed zero, or unknown — from a stat-only
    walk that discloses it computed no hashes. Store bookkeeping is not a rescued file,
    an empty readable manifest alone never proves zero (the walk does), an unreadable
    store is unknown (never zero), and a split root walks the child-drive store too."""
    from ouroboros.headless import task_artifacts_dir

    drive_logs = tmp_path / "logs"
    drive_logs.mkdir()

    def fact(task_id, env=None, **task_extra):
        pipeline._record_task_facts(env=env, task={"id": task_id, "chat_id": 1, **task_extra},
                                    usage={"rounds": 1}, llm_trace={"tool_calls": []}, drive_logs=drive_logs)
        [row] = [r for r in _rows(drive_logs) if r["summary_id"] == f"task-facts:{task_id}"]
        return row["files_rescued"]

    store = task_artifacts_dir(tmp_path, "pos-1")
    (store / "report.md").write_text("r", encoding="utf-8")
    (store / "nested").mkdir()
    (store / "nested" / "data.csv").write_text("1,2", encoding="utf-8")
    (store / ".artifact_manifest.json").write_text("{}", encoding="utf-8")
    (store / ".scratch_manifest.json").write_text("{}", encoding="utf-8")
    assert fact("pos-1") == {"count": 2, "state": "positive", "hash_computed": False,
                             "stores": [{"store": str(store), "count": 2, "readable": True}]}

    store = task_artifacts_dir(tmp_path, "zero-1")
    (store / ".artifact_manifest.json").write_text('{"schema_version": 1, "artifacts": {}}', encoding="utf-8")
    assert fact("zero-1") == {"count": 0, "state": "zero", "hash_computed": False,
                              "stores": [{"store": str(store), "count": 0, "readable": True}]}
    never_created = task_artifacts_dir(tmp_path, "none-1", create=False)
    assert fact("none-1")["state"] == "zero" and not never_created.exists()

    blocked = task_artifacts_dir(tmp_path, "unk-1", create=False)
    blocked.write_text("a file where the store directory should be", encoding="utf-8")
    assert fact("unk-1") == {"count": 0, "state": "unknown", "hash_computed": False,
                             "stores": [{"store": str(blocked), "count": 0, "readable": False}]}

    child = tmp_path / "child-drive"
    (task_artifacts_dir(child, "split-1") / "out.txt").write_text("o", encoding="utf-8")
    canonical = task_artifacts_dir(tmp_path, "split-1")
    assert fact("split-1", env=SimpleNamespace(drive_root=child), budget_drive_root=str(tmp_path)) == {
        "count": 1, "state": "positive", "hash_computed": False,
        "stores": [{"store": str(canonical), "count": 0, "readable": True},
                   {"store": str(task_artifacts_dir(child, "split-1", create=False)), "count": 1, "readable": True}]}


def test_rescued_files_walk_excludes_exactly_the_store_bookkeeping_names():
    """The bookkeeping names the walk skips are the SSOT literals, pinned so they cannot drift."""
    from ouroboros.artifacts import _ARTIFACT_MANIFEST
    from ouroboros.task_finalization import RESCUED_FILES_BOOKKEEPING
    from ouroboros.workspace_patch_capture import SCRATCH_MANIFEST_NAME

    assert RESCUED_FILES_BOOKKEEPING == frozenset({_ARTIFACT_MANIFEST, SCRATCH_MANIFEST_NAME})
