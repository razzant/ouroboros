import json
import pathlib
from types import SimpleNamespace

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
        "cost_usd_with_children": 1.25,
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
    assert row["cost_usd_with_children"] == 1.25
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


def test_emit_task_results_queues_restart_after_final_events(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "_store_task_result", lambda *args, **kwargs: None)
    memory_calls = []
    monkeypatch.setattr(pipeline, "_run_chat_consolidation", lambda *args, **kwargs: memory_calls.append("chat"))
    monkeypatch.setattr(pipeline, "_run_scratchpad_consolidation", lambda *args, **kwargs: memory_calls.append("scratchpad"))
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *args, **kwargs: memory_calls.append("post_task"))

    pending_events = []
    ctx = SimpleNamespace(pending_restart_reason="apply timeout fix")
    env = SimpleNamespace(drive_root=tmp_path)
    drive_logs = tmp_path / "logs"
    drive_logs.mkdir(parents=True)

    pipeline.emit_task_results(
        env=env,
        memory=object(),
        llm=object(),
        pending_events=pending_events,
        task={"id": "task-1", "type": "task", "chat_id": 1, "text": "do it"},
        text="All done",
        usage={"rounds": 2, "cost": 0.2},
        llm_trace={"tool_calls": [], "reasoning_notes": []},
        start_time=0.0,
        drive_logs=drive_logs,
        ctx=ctx,
    )

    assert [evt["type"] for evt in pending_events] == [
        "send_message",
        "task_metrics",
        "task_done",
        "restart_request",
    ]
    assert pending_events[-1]["reason"] == "apply timeout fix"
    assert ctx.pending_restart_reason is None
    # Consolidations now run inside the single post-task worker; replacing that
    # worker in this ordering test intentionally replaces the whole phase.
    assert memory_calls == ["post_task"]

    pending_events.clear()
    evolution_ctx = SimpleNamespace(
        pending_restart_reason="apply reviewed evolution",
        pending_restart_is_evolution=True,
    )
    pipeline.emit_task_results(
        env=env,
        memory=object(),
        llm=object(),
        pending_events=pending_events,
        task={"id": "evo-1", "type": "evolution", "chat_id": 1, "text": "improve"},
        text="All done",
        usage={"rounds": 2, "cost": 0.2},
        llm_trace={"tool_calls": [], "reasoning_notes": []},
        start_time=0.0,
        drive_logs=drive_logs,
        ctx=evolution_ctx,
    )
    assert [evt["type"] for evt in pending_events] == ["send_message", "task_metrics", "task_done"]
    assert evolution_ctx.pending_restart_reason is None
    assert evolution_ctx.pending_restart_is_evolution is False

    pending_events.clear()
    memory_calls.clear()
    pipeline.emit_task_results(
        env=env,
        memory=object(),
        llm=object(),
        pending_events=pending_events,
        task={"id": "child-1", "type": "task", "chat_id": 1, "text": "inspect", "delegation_role": "subagent", "memory_mode": "shared"},
        text="summary",
        usage={"rounds": 2, "cost": 0.2},
        llm_trace={"tool_calls": [], "reasoning_notes": []},
        start_time=0.0,
        drive_logs=drive_logs,
        ctx=SimpleNamespace(pending_restart_reason=""),
    )
    assert [evt["type"] for evt in pending_events] == ["send_message", "task_metrics", "task_done"]
    assert memory_calls == []


def test_lineage_child_without_delegation_role_cannot_run_global_post_task(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "_store_task_result", lambda *args, **kwargs: None)
    memory_calls = []
    monkeypatch.setattr(pipeline, "_run_chat_consolidation", lambda *a, **k: memory_calls.append("chat"))
    monkeypatch.setattr(pipeline, "_run_scratchpad_consolidation", lambda *a, **k: memory_calls.append("scratchpad"))
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: memory_calls.append("post_task"))
    drive_logs = tmp_path / "logs-child-lineage"
    drive_logs.mkdir()

    pipeline.emit_task_results(
        env=SimpleNamespace(drive_root=tmp_path),
        memory=object(),
        llm=object(),
        pending_events=[],
        task={
            "id": "child-2",
            "root_task_id": "root-1",
            "parent_task_id": "root-1",
            "type": "task",
            "chat_id": 1,
        },
        text="child result",
        usage={"rounds": 1, "cost": 0.0},
        llm_trace={"tool_calls": [], "reasoning_notes": []},
        start_time=0.0,
        drive_logs=drive_logs,
        ctx=SimpleNamespace(pending_restart_reason=""),
    )
    assert memory_calls == []


def test_split_drive_root_runs_one_canonical_post_task_synthesis(tmp_path, monkeypatch):
    child = tmp_path / "child"
    canonical = tmp_path / "canonical"
    child.mkdir()
    canonical.mkdir()
    (child / "logs").mkdir()
    (canonical / "logs").mkdir()
    monkeypatch.setattr(pipeline, "_store_task_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_run_chat_consolidation", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_run_scratchpad_consolidation", lambda *a, **k: None)
    calls = []

    def fake_post(env, task, *_args, **_kwargs):
        calls.append((pathlib.Path(env.drive_root), task.get("child_drive_root")))
        return {"backlog_candidates": []}

    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", fake_post)
    pipeline.emit_task_results(
        env=SimpleNamespace(repo_dir=tmp_path, drive_root=child),
        memory=object(),
        llm=object(),
        pending_events=[],
        task={
            "id": "root-split",
            "root_task_id": "root-split",
            "type": "task",
            "chat_id": 1,
            "budget_drive_root": str(canonical),
        },
        text="done",
        usage={"rounds": 2, "cost": 0.1},
        llm_trace={"tool_calls": [], "reasoning_notes": []},
        start_time=0.0,
        drive_logs=child / "logs",
        ctx=SimpleNamespace(pending_restart_reason=""),
    )
    assert calls == [(canonical, str(child))]


def test_root_phase_checkpoint_is_durable_and_completion_is_idempotent(tmp_path):
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    task = {"id": "root-checkpoint", "root_task_id": "root-checkpoint", "type": "task"}
    trace = {
        "tool_calls": [],
        "reasoning_notes": [],
        "root_phase_checkpoint": {
            "phase": "task_acceptance",
            "status": "pass",
            "pass_index": 1,
            "post_task_synthesis": "pending_once",
        },
    }
    pipeline._store_task_result(
        env, task, "done", {"rounds": 1, "cost": 0.0}, trace,
    )
    stored = pipeline.load_task_result(tmp_path, "root-checkpoint")
    assert stored["root_phase_checkpoint"]["post_task_synthesis"] == "pending_once"
    pipeline._set_root_post_task_checkpoint(env, task, "completed")
    assert pipeline._root_post_task_already_completed(env, task) is True

    # A repeated result materialization must preserve the terminal phase marker.
    pipeline._store_task_result(
        env, task, "done again", {"rounds": 1, "cost": 0.0}, trace,
    )
    stored = pipeline.load_task_result(tmp_path, "root-checkpoint")
    assert stored["root_phase_checkpoint"]["post_task_synthesis"] == "completed"

    degraded_task = {"id": "root-degraded", "root_task_id": "root-degraded"}
    pipeline.write_task_result(
        tmp_path, "root-degraded", pipeline.STATUS_COMPLETED,
        root_phase_checkpoint={"post_task_synthesis": "degraded"},
    )
    assert pipeline._root_post_task_already_completed(env, degraded_task) is True


def test_root_checkpoint_reconciles_exact_subtree_and_late_namer_cost(tmp_path):
    from ouroboros import usage_accounting as accounting

    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    task = {
        "id": "root-cost", "root_task_id": "root-cost", "type": "task",
        "budget_drive_root": str(tmp_path),
    }
    pipeline.write_task_result(
        tmp_path, "root-cost", pipeline.STATUS_COMPLETED,
        root_task_id="root-cost", cost_usd=99.0, cost_final=True,
        root_phase_checkpoint={"post_task_synthesis": "running"},
    )

    def settle(task_id, cost):
        reservation = accounting.reserve_attempt(accounting.AttemptRequest(
            model="openai/gpt-5.2", provider="openai", reservation_usd=cost,
            drive_root=tmp_path, task_id=task_id, root_task_id="root-cost",
            global_limit_usd=10.0, root_limit_usd=10.0,
        ))
        accounting.mark_dispatched(reservation)
        accounting.settle_attempt(reservation, {}, cost_usd=cost, cost_final=True)

    settle("root-cost", 1.0)
    settle("abnormal-child", 2.0)
    pipeline._set_root_post_task_checkpoint(env, task, "completed")
    stored = pipeline.load_task_result(tmp_path, "root-cost")
    assert stored["cost_usd"] == 1.0
    assert stored["cost_usd_with_children"] == 3.0
    assert stored["cost_final"] is True
    assert stored["cost_with_children_partial"] is False

    settle("root-cost", 0.25)
    pipeline._set_root_post_task_checkpoint(env, task, "refresh")
    stored = pipeline.load_task_result(tmp_path, "root-cost")
    assert stored["root_phase_checkpoint"]["post_task_synthesis"] == "completed"
    assert stored["cost_usd"] == 1.25
    assert stored["cost_usd_with_children"] == 3.25


def test_retry_root_checkpoint_preserves_logical_subtree_cost(tmp_path):
    from ouroboros import usage_accounting as accounting

    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    task = {
        "id": "retry-2",
        "root_task_id": "logical-root",
        "parent_task_id": "",
        "delegation_role": "root",
        "original_task_id": "retry-1",
        "timeout_retry_from": "retry-1",
        "budget_drive_root": str(tmp_path),
    }
    assert pipeline._is_root_post_task(task) is True
    assert pipeline._is_root_post_task({
        **task,
        "timeout_retry_from": "different-attempt",
    }) is False
    pipeline.write_task_result(
        tmp_path,
        "retry-2",
        pipeline.STATUS_COMPLETED,
        **{key: value for key, value in task.items() if key != "id"},
        root_phase_checkpoint={"post_task_synthesis": "running"},
    )

    def settle(task_id, cost):
        reservation = accounting.reserve_attempt(accounting.AttemptRequest(
            model="openai/gpt-5.2",
            provider="openai",
            reservation_usd=cost,
            drive_root=tmp_path,
            task_id=task_id,
            root_task_id="logical-root",
            global_limit_usd=10.0,
            root_limit_usd=10.0,
        ))
        accounting.mark_dispatched(reservation)
        accounting.settle_attempt(
            reservation, {}, cost_usd=cost, cost_final=True,
        )

    settle("logical-root", 1.25)
    settle("retry-2", 0.75)
    pipeline._set_root_post_task_checkpoint(env, task, "completed")

    stored = pipeline.load_task_result(tmp_path, "retry-2")
    assert stored["root_task_id"] == "logical-root"
    assert stored["cost_usd"] == 0.75
    assert stored["cost_usd_with_children"] == 2.0
    assert stored["cost_final"] is True


def test_startup_recovery_reuses_pending_root_result_checkpoint(tmp_path, monkeypatch):
    pipeline.write_task_result(
        tmp_path,
        "recover-root",
        pipeline.STATUS_COMPLETED,
        root_task_id="recover-root",
        objective="finish recovery",
        total_rounds=3,
        cost_usd=0.25,
        root_phase_checkpoint={
            "phase": "task_acceptance",
            "status": "pass",
            "post_task_synthesis": "pending_once",
        },
    )
    calls = []

    def fake_run(env, task, usage, trace, evidence, drive_logs, *, blocking=False):
        calls.append((env.drive_root, task, usage, trace, evidence, drive_logs, blocking))
        pipeline._set_root_post_task_checkpoint(env, task, "completed")

    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", fake_run)
    assert pipeline.recover_pending_root_post_task_synthesis(tmp_path, tmp_path / "repo") == 1
    assert calls[0][1]["id"] == "recover-root"
    assert calls[0][2]["rounds"] == 3
    assert calls[0][3]["recovered_post_task_synthesis"] is True
    assert calls[0][-1] is False
    assert pipeline.recover_pending_root_post_task_synthesis(tmp_path, tmp_path / "repo") == 0


def test_startup_recovery_never_replays_indeterminate_paid_post_task_phase(tmp_path, monkeypatch):
    pipeline.write_task_result(
        tmp_path,
        "crashed-root",
        pipeline.STATUS_COMPLETED,
        root_task_id="crashed-root",
        root_phase_checkpoint={
            "phase": "task_acceptance",
            "status": "pass",
            "post_task_synthesis": "running",
        },
    )
    paid_replays = []
    monkeypatch.setattr(
        pipeline,
        "_run_post_task_processing_async",
        lambda *args, **kwargs: paid_replays.append((args, kwargs)),
    )

    assert pipeline.recover_pending_root_post_task_synthesis(tmp_path, tmp_path / "repo") == 1
    assert paid_replays == []
    stored = pipeline.load_task_result(tmp_path, "crashed-root")
    checkpoint = stored["root_phase_checkpoint"]
    assert checkpoint["post_task_synthesis"] == "degraded"
    assert checkpoint["post_task_stop_reason"] == "restart_indeterminate_running"
    assert pipeline.recover_pending_root_post_task_synthesis(tmp_path, tmp_path / "repo") == 0


def test_periodic_orphan_reconcile_does_not_degrade_live_post_task_synthesis(tmp_path):
    from ouroboros.task_status import reconcile_orphaned_running_tasks

    pipeline.write_task_result(
        tmp_path,
        "live-synthesis",
        pipeline.STATUS_COMPLETED,
        root_task_id="live-synthesis",
        root_phase_checkpoint={
            "phase": "task_acceptance",
            "status": "pass",
            "post_task_synthesis": "running",
        },
    )

    assert reconcile_orphaned_running_tasks(tmp_path) == 0
    stored = pipeline.load_task_result(tmp_path, "live-synthesis")
    assert stored["root_phase_checkpoint"]["post_task_synthesis"] == "running"


def test_task_result_and_task_done_mirror_authoritative_review_status(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *args, **kwargs: None)
    pending_events = []
    trace = {
        "tool_calls": [],
        "reasoning_notes": [],
        "review_decision": {"eligibility": "eligible", "trigger": "review_run"},
        "review_runs": [{
            "authority": "host_root",
            "aggregate_signal": "PASS",
            "actors": [{
                "signal": "PASS",
                "parsed": {"outcome_tier": "solved"},
            }],
        }],
    }
    drive_logs = tmp_path / "logs"
    drive_logs.mkdir()

    pipeline.emit_task_results(
        env=SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path),
        memory=object(),
        llm=object(),
        pending_events=pending_events,
        task={
            "id": "review-mirror",
            "root_task_id": "review-mirror",
            "type": "task",
            "chat_id": 1,
            "text": "verify",
        },
        text="done",
        usage={"rounds": 1, "cost": 0.0},
        llm_trace=trace,
        start_time=0.0,
        drive_logs=drive_logs,
        ctx=SimpleNamespace(pending_restart_reason=""),
    )

    stored = pipeline.load_task_result(tmp_path, "review-mirror")
    assert stored["review_status"] == stored["outcome_axes"]["review"]
    assert stored["review_status"]["status"] == "pass"
    done = next(row for row in pending_events if row["type"] == "task_done")
    assert done["review_status"] == stored["review_status"]
    # The durable elapsed scalar is the SAME value the task_metrics event carries,
    # never an independently re-derived second timing.
    metrics = next(row for row in pending_events if row["type"] == "task_metrics")
    assert stored["duration_sec"] == metrics["duration_sec"]


def test_emit_task_results_ephemeral_turn_skips_all_durable_memory(tmp_path, monkeypatch):
    """WS10 idempotency contract (claudexor B5): an ephemeral same-route turn must
    write NO durable memory — not chat/scratchpad consolidation, not reflection/
    evolution — while still delivering its reply."""
    store_calls = []
    monkeypatch.setattr(pipeline, "_store_task_result", lambda *args, **kwargs: store_calls.append(1))
    memory_calls = []
    monkeypatch.setattr(pipeline, "_run_chat_consolidation", lambda *args, **kwargs: memory_calls.append("chat"))
    monkeypatch.setattr(pipeline, "_run_scratchpad_consolidation", lambda *args, **kwargs: memory_calls.append("scratchpad"))
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *args, **kwargs: memory_calls.append("post_task"))

    pending_events = []
    drive_logs = tmp_path / "logs2"
    drive_logs.mkdir(parents=True)
    pipeline.emit_task_results(
        env=SimpleNamespace(drive_root=tmp_path),
        memory=object(),
        llm=object(),
        pending_events=pending_events,
        task={"id": "eph-1", "type": "task", "chat_id": 1, "text": "2+2?", "_is_direct_chat": True, "_ephemeral_turn": True},
        text="4",
        usage={"rounds": 1, "cost": 0.01},
        llm_trace={"tool_calls": [], "reasoning_notes": []},
        start_time=0.0,
        drive_logs=drive_logs,
        ctx=SimpleNamespace(pending_restart_reason=""),
    )
    assert "send_message" in [evt["type"] for evt in pending_events]  # reply still delivered
    inline = next(evt for evt in pending_events if evt["type"] == "send_message")
    assert inline["progress_meta"] == {"ephemeral_decision": True}
    assert memory_calls == []  # NO durable memory writes for an ephemeral turn
    assert store_calls == []  # CW3: no durable task_result for a transient decision turn
    # CW3: task_done carries _ephemeral so the supervisor handler skips the missing-result fallback.
    done = next(evt for evt in pending_events if evt["type"] == "task_done")
    assert done.get("_ephemeral") is True
    assert done.get("ephemeral_decision") is True


def test_ephemeral_typed_routing_delivers_nonempty_final_and_keeps_receipt_metadata(tmp_path, monkeypatch):
    """A typed receipt annotates the owner message; normalized final model prose
    remains one durable assistant reply for every routing action."""
    monkeypatch.setattr(pipeline, "_store_task_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_run_chat_consolidation", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_run_scratchpad_consolidation", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *args, **kwargs: None)
    drive_logs = tmp_path / "routing-logs"
    drive_logs.mkdir(parents=True)

    for action in (
        "route_to_project",
        "steer_task",
        "promote_chat_to_task",
        "routing_manual_target",
    ):
        pending_events = []
        pipeline.emit_task_results(
            env=SimpleNamespace(drive_root=tmp_path),
            memory=object(),
            llm=object(),
            pending_events=pending_events,
            task={
                "id": f"eph-{action}",
                "type": "task",
                "chat_id": 1,
                "text": "route this",
                "_is_direct_chat": True,
                "_ephemeral_turn": True,
            },
            text=f"Receipt prose for {action}",
            usage={"rounds": 1, "cost": 0.01},
            llm_trace={"tool_calls": [{"tool": action}], "reasoning_notes": []},
            start_time=0.0,
            drive_logs=drive_logs,
            ctx=SimpleNamespace(
                pending_restart_reason="",
                _typed_routing_action_emitted=action,
            ),
        )
        sends = [evt for evt in pending_events if evt["type"] == "send_message"]
        assert len(sends) == 1
        assert sends[0]["text"] == f"Receipt prose for {action}"
        assert sends[0]["log_text"] == f"Receipt prose for {action}"
        assert sends[0]["progress_meta"] == {"ephemeral_decision": True}
        done = next(evt for evt in pending_events if evt["type"] == "task_done")
        assert pending_events.index(sends[0]) < pending_events.index(done)
        assert done["ephemeral_decision"] is True
        assert done["typed_routing_action"] == action


def test_project_scoped_post_task_processing_feeds_global_backlog_but_project_memory(tmp_path, monkeypatch):
    import ouroboros.post_task_evolution as post_task_evolution

    calls = []
    reflection = {"backlog_candidates": [{"summary": "tool friction"}], "memory_actions": [{"kind": "note"}]}
    monkeypatch.setattr(pipeline, "_run_task_summary", lambda *args, **kwargs: calls.append(("summary",)))
    monkeypatch.setattr(pipeline, "_run_reflection", lambda *args, **kwargs: reflection)
    monkeypatch.setattr(pipeline, "_update_improvement_backlog", lambda _env, entry: calls.append(("backlog", entry)) or 1)
    monkeypatch.setattr(
        pipeline,
        "_apply_reflection_memory_actions",
        lambda _env, entry, project_id="": calls.append(("memory", project_id, entry)) or 1,
    )
    monkeypatch.setattr(post_task_evolution, "maybe_promote", lambda _env, task, entry, _llm: calls.append(("promote", task.get("project_id"), entry)))
    env = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, drive_path=lambda rel: tmp_path / rel)

    pipeline._run_post_task_processing_async(
        env,
        {"id": "task-1", "type": "task", "project_id": "proj-1", "text": "fix workspace"},
        {"rounds": 3, "cost": 0.1},
        {"tool_calls": [], "reasoning_notes": []},
        {},
        tmp_path / "logs",
        blocking=True,
    )

    assert ("backlog", reflection) in calls
    assert ("memory", "proj-1", reflection) in calls
    assert ("promote", "proj-1", reflection) in calls


def test_root_synthesis_uses_one_shared_nonfinal_subtree_cost_snapshot(tmp_path, monkeypatch):
    import ouroboros.memory as memory_mod
    import ouroboros.post_task_evolution as post_task_evolution
    import ouroboros.usage_accounting as accounting
    import ouroboros.llm as llm_mod

    reads = []
    order = []
    snapshots = []

    def fake_breakdown(root, *, root_task_id="", task_id=""):
        order.append("snapshot")
        reads.append((root, root_task_id, task_id))
        return {
            "accounted_usd": 4.75,
            "reserved_usd": 1.5,
            "unresolved_upper_bound_usd": 0.75,
            "unknown_unmetered": 2,
            "integrity_degraded": False,
        }

    monkeypatch.setattr(accounting, "usage_breakdown", fake_breakdown)
    monkeypatch.setattr(llm_mod, "LLMClient", lambda: object())
    monkeypatch.setattr(memory_mod, "Memory", lambda **_kwargs: object())
    monkeypatch.setattr(
        pipeline, "_run_chat_consolidation",
        lambda *args, **kwargs: order.append("chat_consolidation"),
    )
    monkeypatch.setattr(
        pipeline, "_run_scratchpad_consolidation",
        lambda *args, **kwargs: order.append("scratchpad_consolidation"),
    )
    monkeypatch.setattr(
        pipeline,
        "_run_task_summary",
        lambda _env, _llm, _task, usage, *_args, **_kwargs: (
            order.append("summary"), snapshots.append(usage)
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_run_reflection",
        lambda _env, _llm, _task, usage, *_args, **_kwargs: (
            order.append("reflection"), snapshots.append(usage)
        ),
    )
    monkeypatch.setattr(pipeline, "_update_improvement_backlog", lambda *args, **kwargs: 0)
    monkeypatch.setattr(pipeline, "_apply_reflection_memory_actions", lambda *args, **kwargs: 0)
    monkeypatch.setattr(post_task_evolution, "maybe_promote", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_set_root_post_task_checkpoint", lambda *args, **kwargs: None)

    env = SimpleNamespace(
        repo_dir=tmp_path,
        drive_root=tmp_path,
        drive_path=lambda rel: tmp_path / rel,
    )
    pipeline._run_post_task_processing_async(
        env,
        {
            "id": "root-synthesis",
            "root_task_id": "root-synthesis",
            "budget_drive_root": str(tmp_path),
        },
        {"rounds": 8, "cost": 1.25},
        {"tool_calls": [], "reasoning_notes": []},
        {},
        tmp_path / "logs",
        blocking=True,
    )

    assert reads == [(tmp_path, "root-synthesis", "")]
    assert order[:5] == [
        "snapshot", "chat_consolidation", "scratchpad_consolidation",
        "summary", "reflection",
    ]
    assert len(snapshots) == 2 and snapshots[0] is snapshots[1]
    snapshot = snapshots[0]
    assert snapshot["cost_usd_with_children"] == 4.75
    assert snapshot["reserved_usd"] == 1.5
    assert snapshot["unresolved_upper_bound_usd"] == 0.75
    assert snapshot["unknown_unmetered"] == 2
    assert snapshot["ledger_integrity"] == "ok"
    assert snapshot["cost_final"] is False
    assert snapshot["cost_with_children_partial"] is True


def test_nonblocking_post_task_snapshot_precedes_worker_dispatch(tmp_path, monkeypatch):
    import ouroboros.usage_accounting as accounting

    order = []
    worker_targets = []

    monkeypatch.setattr(
        accounting,
        "usage_breakdown",
        lambda *_args, **_kwargs: order.append("snapshot") or {
            "accounted_usd": 1.0,
            "reserved_usd": 0.0,
            "unresolved_upper_bound_usd": 0.0,
            "unknown_unmetered": 0,
            "integrity_degraded": False,
        },
    )
    monkeypatch.setattr(pipeline, "_set_root_post_task_checkpoint", lambda *args, **kwargs: None)

    class DeferredThread:
        def __init__(self, *, target, daemon):
            assert order == ["snapshot"]
            assert daemon is True
            worker_targets.append(target)

        def start(self):
            order.append("thread_start")

    monkeypatch.setattr(pipeline.threading, "Thread", DeferredThread)
    env = SimpleNamespace(
        repo_dir=tmp_path,
        drive_root=tmp_path,
        drive_path=lambda rel: tmp_path / rel,
    )

    pipeline._run_post_task_processing_async(
        env,
        {
            "id": "async-root",
            "root_task_id": "async-root",
            "budget_drive_root": str(tmp_path),
        },
        {"cost": 0.5},
        {},
        {},
        tmp_path / "logs",
    )

    assert order == ["snapshot", "thread_start"]
    assert len(worker_targets) == 1
    with pipeline._POST_TASK_SYNTHESIS_LOCK:
        pipeline._POST_TASK_SYNTHESIS_INFLIGHT.discard(
            (str(tmp_path.resolve(strict=False)), "async-root")
        )


def test_pre_synthesis_cost_failure_is_unavailable_not_zero(tmp_path, monkeypatch):
    import ouroboros.usage_accounting as accounting

    monkeypatch.setattr(
        accounting,
        "usage_breakdown",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("ledger unavailable")),
    )
    env = SimpleNamespace(drive_root=tmp_path)
    snapshot = pipeline._pre_synthesis_usage_snapshot(
        env,
        {"id": "root", "root_task_id": "root", "budget_drive_root": str(tmp_path)},
        {"rounds": 2, "cost": 1.0},
    )

    assert snapshot["cost_usd_with_children"] is None
    assert snapshot["reserved_usd"] is None
    assert snapshot["unresolved_upper_bound_usd"] is None
    assert snapshot["unknown_unmetered"] is None
    assert snapshot["ledger_integrity"] == "unavailable"
    assert pipeline._synthesis_cost_text(snapshot) == "cost unavailable (non-final)"


def _capture_summary_and_reflection_prompts(
    tmp_path, monkeypatch, usage, *, task_overrides=None,
):
    import ouroboros.consolidator as consolidator

    monkeypatch.setattr(
        consolidator,
        "_consolidation_route",
        lambda: ("test/synthesis-model", False),
    )

    class CapturingLlm:
        def __init__(self):
            self.prompts = []

        def chat(self, *, messages, **_kwargs):
            self.prompts.append(messages[0]["content"])
            return {"content": "captured synthesis"}, {}

    drive_logs = tmp_path / "logs"
    drive_logs.mkdir(parents=True, exist_ok=True)
    task = {
        "id": "root-synthesis-prompt",
        "root_task_id": "root-synthesis-prompt",
        "type": "task",
        "text": "Inspect the shared cost snapshot",
        "drive_root": str(tmp_path),
    }
    task.update(task_overrides or {})
    trace = {
        "tool_calls": [{
            "tool": "run_command",
            "status": "error",
            "is_error": True,
            "result": "TOOL_ERROR: synthetic prompt-capture trigger",
        }],
        "reasoning_notes": [],
    }

    summary_llm = CapturingLlm()
    pipeline._run_task_summary(
        env=None,
        llm=summary_llm,
        task=task,
        usage=usage,
        llm_trace=trace,
        drive_logs=drive_logs,
    )

    reflection_llm = CapturingLlm()
    entry = pipeline._run_reflection(
        SimpleNamespace(drive_root=tmp_path),
        reflection_llm,
        task,
        usage,
        trace,
        {},
    )

    assert entry is not None
    assert len(summary_llm.prompts) == 1
    assert len(reflection_llm.prompts) == 1
    return summary_llm.prompts[0], reflection_llm.prompts[0]


def test_shared_cost_snapshot_reaches_summary_and_reflection_prompts(tmp_path, monkeypatch):
    snapshot = {
        "rounds": 8,
        "cost": 1.25,
        "cost_usd_with_children": 4.75,
        "reserved_usd": 1.5,
        "unresolved_upper_bound_usd": 0.75,
        "unknown_unmetered": 2,
        "ledger_integrity": "ok",
        "cost_snapshot_at": "2026-07-15T12:34:56+00:00",
        "cost_final": False,
        "cost_with_children_partial": True,
        "cost_accounting_status": "available",
        "reason_code": "child_results_deferred",
        "outcome_axes": {
            "execution": {"status": "degraded"},
            "objective": {"status": "best_effort"},
            "review": {"status": "degraded"},
        },
    }

    prompts = _capture_summary_and_reflection_prompts(
        tmp_path, monkeypatch, snapshot,
    )
    snapshot_text = pipeline._synthesis_usage_snapshot_text(snapshot)
    expected_fragments = (
        '"cost_usd_with_children": 4.75',
        '"reserved_usd": 1.5',
        '"unresolved_upper_bound_usd": 0.75',
        '"unknown_unmetered": 2',
        '"ledger_integrity": "ok"',
        '"cost_snapshot_at": "2026-07-15T12:34:56+00:00"',
        '"cost_final": false',
        '"cost_with_children_partial": true',
        '"cost_accounting_status": "available"',
        '"reason_code": "child_results_deferred"',
        '"status": "best_effort"',
    )
    for prompt in prompts:
        assert snapshot_text in prompt
        assert "accounted subtree cost only" in prompt
        assert "separate non-final exposure fields" in prompt
        assert "including the reserved" not in prompt
        assert "outcome_axes` is canonical task truth" in prompt
        assert '"review": {' in prompt
        for fragment in expected_fragments:
            assert fragment in prompt


def test_unavailable_cost_snapshot_is_null_not_zero_in_both_prompts(tmp_path, monkeypatch):
    snapshot = {
        "rounds": 8,
        "cost": 1.25,
        "cost_usd_with_children": None,
        "reserved_usd": None,
        "unresolved_upper_bound_usd": None,
        "unknown_unmetered": None,
        "ledger_integrity": "unavailable",
        "cost_snapshot_at": "2026-07-15T12:35:00+00:00",
        "cost_final": False,
        "cost_with_children_partial": True,
        "cost_accounting_status": "unavailable",
    }

    prompts = _capture_summary_and_reflection_prompts(
        tmp_path, monkeypatch, snapshot,
    )
    snapshot_text = pipeline._synthesis_usage_snapshot_text(snapshot)
    null_fields = (
        "cost_usd_with_children",
        "reserved_usd",
        "unresolved_upper_bound_usd",
        "unknown_unmetered",
    )
    for prompt in prompts:
        assert snapshot_text in prompt
        for field in null_fields:
            assert f'"{field}": null' in prompt
        assert '"ledger_integrity": "unavailable"' in prompt
        assert '"cost_snapshot_at": "2026-07-15T12:35:00+00:00"' in prompt
        assert '"cost_final": false' in prompt
        assert '"cost_with_children_partial": true' in prompt
        assert '"cost_accounting_status": "unavailable"' in prompt
        assert "$0" not in prompt


def test_child_legacy_usage_does_not_claim_a_subtree_snapshot(tmp_path, monkeypatch):
    prompts = _capture_summary_and_reflection_prompts(
        tmp_path,
        monkeypatch,
        {"rounds": 8, "cost": 1.25},
        task_overrides={
            "id": "child-synthesis-prompt",
            "root_task_id": "root-synthesis-prompt",
            "parent_task_id": "root-synthesis-prompt",
            "delegation_role": "subagent",
        },
    )

    for prompt in prompts:
        assert "Shared pre-synthesis cost snapshot" not in prompt
        assert "cost_usd_with_children" not in prompt
        assert "cost_snapshot_at" not in prompt
    assert "Cost: $1.25" in prompts[0]


def test_emit_project_scoped_parent_drive_gets_only_global_backlog_channel(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "_store_task_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "load_task_result", lambda *args, **kwargs: {})
    monkeypatch.setattr(pipeline, "_run_chat_consolidation", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_run_scratchpad_consolidation", lambda *args, **kwargs: None)

    parent = tmp_path / "parent"
    child = tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    reflection = {"backlog_candidates": [{"summary": "workspace tool friction"}], "memory_actions": [{"kind": "note"}]}
    post_calls = []
    global_calls = []

    def fake_post(env, task, *_args, **kwargs):
        post_calls.append((pathlib.Path(env.drive_root), task.get("project_id")))
        callback = kwargs.get("on_reflection")
        if callback is not None:
            callback(reflection, object())
        return reflection

    def fake_global(env, task, entry, _llm):
        global_calls.append((pathlib.Path(env.drive_root), task.get("project_id"), entry))

    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", fake_post)
    monkeypatch.setattr(pipeline, "_run_global_backlog_promotion_only", fake_global)

    pending_events = []
    env = SimpleNamespace(repo_dir=tmp_path, drive_root=child, drive_path=lambda rel: child / rel)
    pipeline.emit_task_results(
        env=env,
        memory=object(),
        llm=object(),
        pending_events=pending_events,
        task={
            "id": "task-project",
            "type": "task",
            "chat_id": 1,
            "text": "fix workspace",
            "project_id": "proj-1",
            "budget_drive_root": str(parent),
        },
        text="Done",
        usage={"rounds": 2, "cost": 0.2},
        llm_trace={"tool_calls": [], "reasoning_notes": []},
        start_time=0.0,
        drive_logs=child / "logs",
        ctx=SimpleNamespace(pending_restart_reason=""),
    )

    assert post_calls == [(child, "proj-1")]
    assert global_calls == [(parent, "proj-1", reflection)]


def test_project_global_promotion_uses_real_maybe_promote_without_project_scope(tmp_path, monkeypatch):
    import ouroboros.post_task_evolution as post_task_evolution

    monkeypatch.setattr("ouroboros.config.get_post_task_evolution_enabled", lambda: True)
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "pro")
    monkeypatch.setattr("ouroboros.config.get_post_task_evolution_cadence", lambda: "every_n:1")
    monkeypatch.setattr(
        post_task_evolution,
        "_decide_promotion",
        lambda *_args, **_kwargs: {
            "promote": True,
            "objective": "Improve Ouroboros workspace tool feedback",
            "requires_plan_review": True,
            "backlog_id": "",
        },
    )
    env = SimpleNamespace(drive_root=tmp_path, drive_path=lambda rel: tmp_path / rel)
    reflection = {
        "reflection": "Project-specific detail should not be forwarded.",
        "memory_actions": [{"kind": "note"}],
        "backlog_candidates": [{"summary": "Improve Ouroboros workspace tool feedback"}],
    }

    pipeline._run_global_backlog_promotion_only(
        env,
        {
            "id": "task-project",
            "project_id": "proj-1",
            "workspace_root": "/tmp/project",
            "workspace_mode": "external",
            "metadata": {"workspace_preflight": {"git": {"head": "abc"}}},
        },
        reflection,
        object(),
    )

    req = json.loads((tmp_path / "state" / "post_task_evolution_request.json").read_text(encoding="utf-8"))
    assert req["objective"] == "Improve Ouroboros workspace tool feedback"
    backlog = (tmp_path / "memory" / "knowledge" / "improvement-backlog.md").read_text(encoding="utf-8")
    assert "Project-specific detail" not in backlog


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

    payload = json.loads((tmp_path / "task_results" / "task-store.json").read_text(encoding="utf-8"))
    assert payload["review_evidence"]["has_evidence"] is True
    assert payload["review_evidence"]["open_obligations"][0]["item"] == "tests_affected"


def test_store_task_result_persists_duration_sec_on_completed_and_failed(tmp_path):
    """Elapsed time must be durable, not only a live ``task_metrics`` event field.

    An owner surface reading ``task_results/<id>.json`` (the task inspector footer)
    can otherwise never show elapsed honestly for a task whose event stream it never
    saw. Both terminal paths persist the SAME scalar ``emit_task_results`` computed,
    and a caller without timing omits the key rather than writing a fabricated 0.
    """
    env = SimpleNamespace(drive_root=tmp_path)

    pipeline._store_task_result(
        env=env,
        task={"id": "task-duration-ok", "type": "task", "text": "hi"},
        text="done",
        usage={"rounds": 1, "cost": 0.0},
        llm_trace={"tool_calls": [], "reasoning_notes": []},
        result_fields={"duration_sec": 12.5},
    )
    payload = json.loads((tmp_path / "task_results" / "task-duration-ok.json").read_text(encoding="utf-8"))
    assert payload["status"] == pipeline.STATUS_COMPLETED
    assert payload["duration_sec"] == 12.5

    # Failed path: the pre-existing failed status is preserved AND still timed.
    pipeline.write_task_result(tmp_path, "task-duration-failed", pipeline.STATUS_FAILED)
    pipeline._store_task_result(
        env=env,
        task={"id": "task-duration-failed", "type": "task", "text": "hi"},
        text="broke",
        usage={"rounds": 1, "cost": 0.0},
        llm_trace={"tool_calls": [], "reasoning_notes": []},
        result_fields={"duration_sec": 3.25},
    )
    payload = json.loads((tmp_path / "task_results" / "task-duration-failed.json").read_text(encoding="utf-8"))
    assert payload["status"] == pipeline.STATUS_FAILED
    assert payload["duration_sec"] == 3.25

    pipeline._store_task_result(
        env=env,
        task={"id": "task-duration-absent", "type": "task", "text": "hi"},
        text="done",
        usage={"rounds": 1, "cost": 0.0},
        llm_trace={"tool_calls": [], "reasoning_notes": []},
    )
    payload = json.loads((tmp_path / "task_results" / "task-duration-absent.json").read_text(encoding="utf-8"))
    assert "duration_sec" not in payload


def test_public_task_result_keeps_persisted_duration_sec(tmp_path):
    """The public projection strips only result_status keys — elapsed survives."""
    from ouroboros.outcomes import public_task_result

    env = SimpleNamespace(drive_root=tmp_path)
    pipeline._store_task_result(
        env=env,
        task={"id": "task-duration-public", "type": "task", "text": "hi"},
        text="done",
        usage={"rounds": 1, "cost": 0.0},
        llm_trace={"tool_calls": [], "reasoning_notes": []},
        result_fields={"duration_sec": 7.0},
    )
    stored = pipeline.load_task_result(tmp_path, "task-duration-public")
    assert public_task_result(stored)["duration_sec"] == 7.0


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

    payload = json.loads(
        (tmp_path / "task_results" / "task-review-projection.json").read_text(encoding="utf-8")
    )
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

    payload = json.loads((tmp_path / "task_results" / "task-failed.json").read_text(encoding="utf-8"))
    assert payload["status"] == STATUS_FAILED
    assert payload["result"] == "final failure reply"


def test_store_task_result_marks_unresolved_tool_failure_failed(tmp_path):
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

    payload = json.loads((tmp_path / "task_results" / "task-tool-failed.json").read_text(encoding="utf-8"))
    assert payload["status"] == STATUS_COMPLETED
    assert payload["outcome_axes"]["execution"]["status"] == "degraded"
    assert payload["outcome_axes"]["objective"]["status"] == "not_evaluated"
    assert payload["reason_code"] == "tool_failure"
    assert payload["loop_outcome"]["failure"]["tool_errors"][0]["status"] == "artifact_output_error"


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

    payload = json.loads((tmp_path / "task_results" / "task-tool-recovered.json").read_text(encoding="utf-8"))
    assert payload["status"] == STATUS_COMPLETED
    assert payload["outcome_axes"]["execution"]["status"] == "ok"
    assert payload["outcome_axes"]["objective"]["status"] == "not_evaluated"
    assert payload["loop_outcome"]["failure"] is None


def test_collect_review_evidence_keeps_recent_attempts_task_scoped(tmp_path):
    from ouroboros.review_evidence import collect_review_evidence
    from ouroboros.review_state import AdvisoryReviewState, CommitAttemptRecord, make_repo_key, save_state

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True)
    (repo_dir / ".git").mkdir()

    state = AdvisoryReviewState()
    state.record_attempt(CommitAttemptRecord(
        ts="2026-04-07T10:00:00+00:00",
        commit_message="other task attempt",
        status="blocked",
        repo_key=make_repo_key(repo_dir),
        tool_name="commit_reviewed",
        task_id="task-other",
        attempt=1,
        block_reason="critical_findings",
    ))
    save_state(tmp_path, state)

    evidence = collect_review_evidence(
        tmp_path,
        task_id="task-current",
        repo_dir=repo_dir,
    )

    assert evidence["recent_attempts"] == []


def test_update_improvement_backlog_appends_candidates(tmp_path):
    env = SimpleNamespace(drive_root=tmp_path)

    added = pipeline._update_improvement_backlog(
        env,
        {
            "backlog_candidates": [{
                "summary": "Reduce recurring task friction around REVIEW_BLOCKED",
                "category": "process",
                "source": "execution_reflection",
                "task_id": "task-backlog",
                "evidence": "REVIEW_BLOCKED",
                "context": "The task retried blocked review loops without narrowing scope.",
                "proposed_next_step": "Run plan_task before touching review prompts again.",
            }],
        },
    )

    assert added == 1
    backlog_path = tmp_path / "memory" / "knowledge" / "improvement-backlog.md"
    assert backlog_path.exists()
    text = backlog_path.read_text(encoding="utf-8")
    assert "Reduce recurring task friction around REVIEW_BLOCKED" in text


def test_run_reflection_returns_entry_when_generated(tmp_path):
    captured = {}

    class FakeLlm:
        def chat(self, *, messages, model, reasoning_effort, max_tokens):
            captured["prompt"] = messages[0]["content"]
            return {
                "content": (
                    "Reflection text.\n"
                    "BACKLOG_CANDIDATES_JSON: "
                    "[{\"summary\":\"Reduce recurring task friction around REVIEW_BLOCKED\"," 
                    "\"category\":\"process\"," 
                    "\"source\":\"execution_reflection\"," 
                    "\"evidence\":\"REVIEW_BLOCKED\"}]"
                )
            }, {"cost": 0}

    env = SimpleNamespace(drive_root=tmp_path)
    (tmp_path / "logs").mkdir(parents=True)

    entry = pipeline._run_reflection(
        env,
        FakeLlm(),
        {"id": "task-reflect", "type": "task", "text": "Fix it"},
        {"rounds": 2, "cost": 0.01},
        {"tool_calls": [{"tool": "commit_reviewed", "is_error": False, "result": "⚠️ REVIEW_BLOCKED"}]},
        {"recent_attempts": [], "open_obligations": [{"item": "tests_affected", "reason": "Fix the failing test before commit"}]},
    )

    assert entry is not None
    assert entry["task_id"] == "task-reflect"
    assert entry["reflection"] == "Reflection text."
    assert len(entry["backlog_candidates"]) == 1
    assert entry["backlog_candidates"][0]["summary"] == "Reduce recurring task friction around REVIEW_BLOCKED"


def test_collect_review_evidence_scopes_open_obligations_to_repo(tmp_path):
    from ouroboros.review_evidence import collect_review_evidence
    from ouroboros.review_state import (
        AdvisoryReviewState,
        AdvisoryRunRecord,
        CommitAttemptRecord,
        compute_snapshot_hash,
        make_repo_key,
        save_state,
    )

    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    repo_a.mkdir(parents=True)
    repo_b.mkdir(parents=True)
    (repo_a / ".git").mkdir()
    (repo_b / ".git").mkdir()
    (repo_a / "tracked.py").write_text("print('repo a')\n", encoding="utf-8")
    (repo_b / "tracked.py").write_text("print('repo b')\n", encoding="utf-8")

    repo_a_key = make_repo_key(repo_a)
    repo_b_key = make_repo_key(repo_b)
    state = AdvisoryReviewState()
    state.add_run(AdvisoryRunRecord(
        snapshot_hash=compute_snapshot_hash(repo_a),
        commit_message="repo a ready",
        status="fresh",
        ts="2026-04-07T10:00:00+00:00",
        repo_key=repo_a_key,
    ))
    state.record_attempt(CommitAttemptRecord(
        ts="2026-04-07T10:01:00+00:00",
        commit_message="repo b blocked",
        status="blocked",
        repo_key=repo_b_key,
        tool_name="commit_reviewed",
        task_id="task-b",
        attempt=1,
        block_reason="critical_findings",
        critical_findings=[{
            "item": "foreign_issue",
            "reason": "other repo only",
            "severity": "critical",
            "verdict": "FAIL",
        }],
    ))
    state.last_stale_from_edit_ts = "2026-04-07T10:02:00+00:00"
    state.last_stale_reason = "repo-b mutation"
    state.last_stale_repo_key = repo_b_key
    save_state(tmp_path, state)

    evidence = collect_review_evidence(tmp_path, repo_dir=repo_a)

    assert evidence["current_repo"]["repo_commit_ready"] is True
    assert evidence["current_repo"]["stale_reason"] == ""
    assert evidence["current_repo"]["stale_ts"] == ""
    assert evidence["open_obligations"] == []
    assert evidence["commit_readiness_debts"] == []


def test_collect_review_evidence_includes_commit_readiness_debt(tmp_path):
    from ouroboros.review_evidence import collect_review_evidence
    from ouroboros.review_state import AdvisoryReviewState, CommitAttemptRecord, make_repo_key, save_state

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True)
    (repo_dir / ".git").mkdir()
    (repo_dir / "tracked.py").write_text("print('hi')\n", encoding="utf-8")

    repo_key = make_repo_key(repo_dir)
    state = AdvisoryReviewState()
    for idx, reason in enumerate(["missing tests", "coverage still missing"], start=1):
        state.record_attempt(CommitAttemptRecord(
            ts=f"2026-04-07T10:0{idx}:00+00:00",
            commit_message=f"blocked {idx}",
            status="blocked",
            repo_key=repo_key,
            tool_name="commit_reviewed",
            task_id=f"task-{idx}",
            attempt=idx,
            block_reason="critical_findings",
            critical_findings=[{
                "item": "tests_affected",
                "reason": reason,
                "severity": "critical",
                "verdict": "FAIL",
            }],
            readiness_warnings=["Start retry from review debt."],
        ))
    save_state(tmp_path, state)

    evidence = collect_review_evidence(tmp_path, repo_dir=repo_dir)

    assert evidence["current_repo"]["repo_commit_ready"] is False
    assert len(evidence["commit_readiness_debts"]) >= 1
    assert evidence["commit_readiness_debts"][0]["category"] in {"obligation_repeat", "readiness_warning"}


def test_truncate_with_notice_uses_utils_ssot():
    """_truncate_with_notice in agent_task_pipeline is now truncate_review_artifact from utils.
    Verify it truncates long strings and adds a visible omission note (no silent clipping)."""
    from ouroboros.utils import truncate_review_artifact
    # The alias in agent_task_pipeline should be the same object
    assert pipeline._truncate_with_notice is truncate_review_artifact

    short = "hello"
    assert pipeline._truncate_with_notice(short, 100) == short

    long_text = "x" * 200
    result = pipeline._truncate_with_notice(long_text, 50)
    assert result.startswith("x" * 50)
    assert "50" in result  # omission note mentions limit
    assert len(result) > 50  # note appended, not just raw slice

    # Handles None gracefully
    assert pipeline._truncate_with_notice(None, 10) == ""


def test_emit_task_results_surfaces_receipt_absent_flag_in_event_stream(tmp_path, monkeypatch):
    # Regression: the receipt_absent / expected_output_ungrounded objective-axis flag must reach
    # the task_eval (events.jsonl) and task_metrics (pending_events) monitoring streams — where the
    # day-1 kill-switch metric reads it — not only the stored task_result.json. Previously the flag
    # was applied inside _store_task_result, AFTER the events were already emitted from an un-flagged
    # outcome, so the event stream never saw it.
    captured = {}
    monkeypatch.setattr(pipeline, "_store_task_result", lambda *a, **k: captured.update(k))
    monkeypatch.setattr(pipeline, "_run_chat_consolidation", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_run_scratchpad_consolidation", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: None)

    pending_events = []
    env = SimpleNamespace(drive_root=tmp_path)
    drive_logs = tmp_path / "logs"
    drive_logs.mkdir(parents=True)

    # reviewable effects (commit_reviewed) + empty receipt store -> receipt_absent
    pipeline.emit_task_results(
        env=env, memory=object(), llm=object(),
        pending_events=pending_events,
        task={"id": "flagme", "type": "task", "chat_id": 1, "text": "do it"},
        text="All done",
        usage={"rounds": 2, "cost": 0.2},
        llm_trace={"tool_calls": [{"tool": "commit_reviewed", "status": "ok"}], "reasoning_notes": []},
        start_time=0.0,
        drive_logs=drive_logs,
        ctx=SimpleNamespace(pending_restart_reason=""),
    )

    # task_metrics event (pending_events) carries the flag
    metrics = next(e for e in pending_events if e["type"] == "task_metrics")
    assert metrics["outcome_axes"]["objective"].get("warning") == "receipt_absent"

    # task_eval event (events.jsonl) carries the flag
    events = [json.loads(line) for line in (drive_logs / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    task_eval = next(e for e in events if e.get("type") == "task_eval")
    assert task_eval["outcome_axes"]["objective"].get("warning") == "receipt_absent"

    # single source: the SAME flagged loop_outcome is threaded to _store_task_result (not re-derived)
    assert captured["loop_outcome"]["outcome_axes"]["objective"].get("warning") == "receipt_absent"
