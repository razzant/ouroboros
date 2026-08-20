"""The root post-task synthesis phase of ``ouroboros.agent_task_pipeline``.

Split out of ``tests/test_agent_task_pipeline.py`` when that module was divided
by theme; every moved block is verbatim. Covers the durable
`root_phase_checkpoint` state machine and its exact-subtree cost
reconciliation, startup recovery of pending/indeterminate synthesis, the
shared pre-synthesis usage snapshot taken once before worker dispatch, and
that snapshot reaching (or staying out of) the summary and reflection prompts.
"""

from types import SimpleNamespace

import ouroboros.agent_task_pipeline as pipeline


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

    def fake_run(env, task, usage, trace, evidence, drive_logs, *, blocking=False,
                 sealed_final=None):
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
