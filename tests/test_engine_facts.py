"""Engine-and-pool facts: counts and durations, with each source able to be absent."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from ouroboros.engine_facts import engine_pool_line


def _forbid_daemon(monkeypatch):
    import ouroboros.claudexor_daemon as daemon
    import ouroboros.claudexor_exit_facts as facts

    def boom(*_args, **_kwargs):
        raise AssertionError("daemon read")

    monkeypatch.setattr(daemon.OwnedClaudexorDaemon, "status_dict", boom)
    monkeypatch.setattr(facts, "engine_status_facts", boom)
    monkeypatch.setattr(daemon, "ensure_owned_gateway", boom)


def _iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _seed(root, monkeypatch):
    monkeypatch.setattr("ouroboros.model_slots._main_model", lambda: "claudexor::codex=primary")
    base = int(time.time())
    from ouroboros import usage_store

    with usage_store.hold(root) as txn:
        txn.conn.execute(
            "INSERT INTO summaries (scope, key, rows, settled, confirmed, estimated, reserved, unresolved, "
            "accounted_num, attempt_counts, unknown_unmetered, priced_rows, tracked_nonfinal_rows, "
            "accounting_open_rows, non_final_rows, subscription_sessions, subscription_windows, processing, "
            "prompt_tokens, prompt_tokens_present, completion_tokens, completion_tokens_present, "
            "cached_tokens, cached_tokens_present, cache_write_tokens, cache_write_tokens_present, "
            "physical_calls, prompt_cache_ttls, caps) VALUES "
            "('category', ?, 1, '0','0','0','0','0', 0, '{}', 0,0,0,0,0, 0, '{}', '{}', "
            "0,0,0,0,0,0,0,0,0, '{}', '{}')",
            ("task",),
        )
        rows = [
            ("s1", "settled", "claudexor::codex=primary", base - 10, base - 100, base - 70),
            ("s2", "settled", "claudexor::codex=primary", base - 20, base - 200, base - 110),
            ("r1", "released", "claudexor::codex=primary", base - 30, base - 40, None),
            ("u1", "unresolved", "claudexor::codex=primary", base - 40, base - 50, None),
            ("other", "settled", "other-model", base - 15, base - 80, base - 40),
            ("old", "settled", "claudexor::codex=primary", base - 5000, base - 5100, base - 5060),
        ]
        for seq, (attempt_id, state, model, epoch, started, ended) in enumerate(rows, 1):
            txn.conn.execute(
                "INSERT INTO attempts (attempt_id, revision, seq, seq_first, state, model, category, "
                "ts_dispatched, ts_final, ts_last, ts_last_epoch) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (attempt_id, 1, seq, seq, state, model, "task",
                 _iso(started), _iso(ended) if ended is not None else None,
                 _iso(ended or started), epoch),
            )
    (root / "logs").mkdir(exist_ok=True)
    (root / "logs" / "supervisor.jsonl").write_text(json.dumps({
        "ts": "2026-11-04T12:00:00+00:00", "type": "claudexor_engine_capacity",
        "loop": {"utilization": 0.97, "delay": {"p50Ms": 12, "p99Ms": 80, "samples": 20}},
    }) + "\n", encoding="utf-8")
    (root / "logs" / "server.log").write_text(
        "2026-11-04 12:00:00,000 [INFO] ouroboros.gateway.claudexor_passive: "
        "claudexor_passive_quota_read reads={} read_errors={} "
        "timings_ms={'discovery': 12, 'accounts': 40, 'total': 80}\n",
        encoding="utf-8")
    state = root / "state"
    state.mkdir(exist_ok=True)
    (state / "queue_snapshot.json").write_text(json.dumps({
        "active_worker_count": 4, "worker_total": 8,
        "assignable_idle_workers": 2, "pending_count": 3,
        "running": [], "pending": [],
    }), encoding="utf-8")
    from ouroboros import store
    fresh = datetime.now(timezone.utc).isoformat()
    with store.open_store(root, write=True) as txn:
        txn.conn.execute(
            "INSERT INTO tasks (task_id, status, updated_at) VALUES (?, 'completed', ?)",
            ("done", fresh))
        txn.conn.execute(
            "INSERT INTO tasks (task_id, status, updated_at) VALUES (?, 'completed', ?)",
            ("ancient", "2020-01-01T00:00:00+00:00"))


def test_row_renders_ledger_snapshot_and_stored_engine_without_a_daemon(tmp_path, monkeypatch):
    _forbid_daemon(monkeypatch)
    _seed(tmp_path, monkeypatch)
    line = engine_pool_line(tmp_path)
    assert "primary claudexor::codex=primary last 10 min: 2 settled, 1 released, 1 unresolved" in line
    assert "dispatch→answer median 60.0s, p90 84.0s" in line
    assert "engine: loop utilisation 0.97, delay p50 12ms p99 80ms samples 20, observed 2026-11-04T12:00:00+00:00" in line
    assert "control latency: discovery 0.012s, accounts 0.040s, total 0.080s, observed 2026-11-04 12:00:00,000" in line
    # A document from before the cap and the acting counts were published: each absent count is named.
    assert ("project worker pool: owner cap on acting tasks not observed, total 8, with a task not observed, "
            "acting not observed, waiting on their process not observed, booting not observed, "
            "reaping (incl. booting) not observed, assignable idle 2, pending rows 3; conversation carriers not observed; "
            "memory admission: not read yet") in line
    assert "completions last hour 1" in line
    lowered = line.lower()
    for word in ("busy", "saturated", "healthy", "degraded", "share", "not delivered", "%"):
        assert word not in lowered
    source = Path("ouroboros/engine_facts.py").read_text(encoding="utf-8")
    for name in ("ensure_owned_gateway", "status_dict", "engine_status_facts", "daemon_status"):
        assert name not in source


def test_each_absent_source_says_not_observed(tmp_path, monkeypatch):
    _forbid_daemon(monkeypatch)
    line = engine_pool_line(tmp_path)
    for phrase in (
        "primary binding: not observed",
        "engine: not observed",
        "control latency: not observed",
        "pool: not observed",
        "completions last hour: not observed",
    ):
        assert phrase in line
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "supervisor.jsonl").write_text(json.dumps({
        "ts": "2026-11-04T01:00:00+00:00", "type": "claudexor_engine_capacity",
    }) + "\n", encoding="utf-8")
    assert "engine: not observed since 2026-11-04T01:00:00+00:00" in engine_pool_line(tmp_path)


def test_reader_does_not_create_a_usage_store(tmp_path, monkeypatch):
    _forbid_daemon(monkeypatch)
    engine_pool_line(tmp_path)
    assert not (tmp_path / "state" / "usage.sqlite").exists()


def test_vitals_carry_the_row(tmp_path, monkeypatch):
    from ouroboros.context_runtime_facts import vitals_fact, vitals_line

    _forbid_daemon(monkeypatch)
    fact = vitals_fact(tmp_path, observed_at="2026-11-04T12:00:00+00:00")
    assert "primary binding: not observed" in fact["engine_pool"]
    assert "engine pool" in vitals_line(fact)


def test_schedule_result_carries_the_row_and_still_queues(tmp_path, monkeypatch):
    from tests.test_available_subagents_runtime import _session_row, _settings
    from tests.test_supervisor_route_flow import root_context, schedule

    ctx = root_context(tmp_path, {**_settings(_session_row()), "OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS": "true"}, monkeypatch)
    answer = schedule(ctx)
    assert answer.startswith("Subagent request queued")
    assert "primary" in answer and "engine:" in answer and "pool:" in answer
    assert "tree slots" not in answer
    assert "\nwrite root: none (read-only child)" in answer
    # An acting child's result names its resolved write root and who chose it.
    project = tmp_path / "project"
    project.mkdir()
    assert f"\nwrite root: {project} (caller)" in schedule(ctx, write_surface="external_workspace", write_root=str(project))
    ctx.workspace_root = str(project)
    assert f"\nwrite root: {project} (parent workspace)" in schedule(ctx, write_surface="external_workspace")
    ctx.workspace_root = ""
    minted = schedule(ctx, write_surface="external_workspace")
    assert "(host-minted coop tree) — children write there" in minted and "pool:" in minted


def test_delegate_start_refusal_carries_the_row_and_a_wait_refusal_does_not():
    from ouroboros.tools.delegate import _fail
    from ouroboros.delegate_shared import delegate_payload

    started = delegate_payload(_fail("delegate_start", "daemon_unreachable", "refused"))
    waiting = delegate_payload(_fail("delegate_wait", "daemon_unreachable", "refused"))
    assert "primary" in started["engine_pool"]
    assert "engine_pool" not in waiting


def test_dispatch_block_attaches_the_fact_without_a_status_read(tmp_path, monkeypatch):
    from ouroboros import claudexor_daemon, subagents
    from tests.test_supervisor_route import selected_task

    _forbid_daemon(monkeypatch)
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: type("G", (), {"close": lambda self: None})())
    monkeypatch.setattr(subagents, "route_health", lambda *_a, **_k: ("subscription_window_exhausted", ""))
    task = selected_task()
    task.update(id="child", delegation_role="subagent", task_constraint={}, budget_drive_root=str(tmp_path))
    dispatch = subagents.resolve_subagent_dispatch(task, task_type="task")
    assert dispatch.executor == "blocked"
    assert "primary" in dispatch.availability["engine_pool"]


def test_no_handle_refusal_says_a_fresh_start_is_safe_and_a_handle_does_not(tmp_path):
    from types import SimpleNamespace
    from ouroboros.subagent_bootstrap import append_startup_receipt

    def receipt(startup):
        messages = []
        append_startup_receipt(
            SimpleNamespace(drive_root=tmp_path, budget_drive_root=str(tmp_path)), messages,
            json.dumps({"status": "configured_session_start_wake", "startup": startup}))
        return messages[0]["content"]

    fresh = receipt({"status": "refused", "reason": "work_order_source_channel_unavailable"})
    assert "no run exists" in fresh and "A fresh delegate_start is safe" in fresh
    assert "primary" in fresh
    held = receipt({"status": "refused", "reason": "work_order_source_channel_unavailable", "run_id": "run-1"})
    assert "Physical custody is unresolved" in held and "Do not call delegate_start" in held
    assert "A fresh delegate_start is safe" not in held
    other = receipt({"status": "configured_session_actor_ready"})
    assert "Physical custody is unresolved" in other and "A fresh delegate_start is safe" not in other


def test_slot_note_is_gone():
    import ouroboros.tools.control as control
    import ouroboros.tools.control_scheduling as scheduling

    assert not hasattr(scheduling, "_subagent_slot_note")
    assert not hasattr(control, "_subagent_slot_note")


def test_indexed_attempt_read_is_the_only_attempt_query():
    text = Path("ouroboros/engine_facts.py").read_text(encoding="utf-8")
    assert "INDEXED BY attempts_category_time" in text
    assert "read_usage_records" not in text
    assert "SELECT 1 FROM attempts LIMIT 1" in text
