import json
import pathlib
from types import SimpleNamespace

import pytest

import ouroboros.agent_task_pipeline as pipeline


def test_direct_project_completion_writes_the_real_last_result_pointer(tmp_path, monkeypatch):
    from ouroboros.projects_registry import create_project, get_project

    project = create_project(tmp_path, "room-proof", name="Room proof")
    task = {"id": "native-room-result", "type": "task", "chat_id": project["chat_id"],
            "project_id": project["id"], "text": "Inspect the room", "_is_direct_chat": True}
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: None)
    pending = []
    pipeline.emit_task_results(
        SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path), None, None,
        pending, task, "Room evidence retained.", {"rounds": 1},
        {"tool_calls": [], "reasoning_notes": []}, 0.0, tmp_path / "logs",
    )
    assert get_project(tmp_path, project["id"])["last_task_result_id"] == task["id"]
    assert pipeline.load_task_result(tmp_path, task["id"])["result"] == "Room evidence retained."
    assert not any(event["type"] == "project_digest" for event in pending)


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
        task={
            "id": "child-1", "type": "task", "chat_id": 1, "text": "inspect",
            "delegation_role": "subagent", "memory_mode": "shared",
            "parent_task_id": "parent-1", "root_task_id": "root-1", "role": "critic",
        },
        text="summary",
        usage={"rounds": 2, "cost": 0.2},
        llm_trace={"tool_calls": [], "reasoning_notes": []},
        start_time=0.0,
        drive_logs=drive_logs,
        ctx=SimpleNamespace(pending_restart_reason=""),
    )
    assert [evt["type"] for evt in pending_events] == ["send_message", "task_metrics", "task_done"]
    assert pending_events[0]["progress_meta"] == {
        "subagent_task_id": "child-1",
        "root_task_id": "root-1",
        "parent_task_id": "parent-1",
        "delegation_role": "subagent",
        "subagent_role": "critic",
        "write_surface": "",
        "task_group_id": "",
        "model_lane": "",
        "effective_model_lane": "",
        "model": "",
        "executor_route": "",
    }
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




def test_direct_typed_routing_delivers_nonempty_final_and_keeps_receipt_metadata(tmp_path, monkeypatch):
    """A typed receipt annotates the owner message; normalized final model prose
    remains one durable assistant reply for every routing action."""
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
                "id": f"direct-{action}",
                "type": "task",
                "chat_id": 1,
                "text": "route this",
                "_is_direct_chat": True,
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
        done = next(evt for evt in pending_events if evt["type"] == "task_done")
        assert "ephemeral_decision" not in sends[0].get("progress_meta", {})
        stored = pipeline.load_task_result(tmp_path, f"direct-{action}")
        assert stored["status"] == "completed"
        assert stored["_is_direct_chat"] is True
        assert done["status"] == stored["status"]
        done = next(evt for evt in pending_events if evt["type"] == "task_done")
        assert pending_events.index(sends[0]) < pending_events.index(done)
        assert "ephemeral_decision" not in done
        assert "_ephemeral" not in done
        assert done["typed_routing_action"] == action


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


def test_stopped_direct_turn_pays_no_post_task_synthesis(tmp_path, monkeypatch):
    """"Stop now" on a direct-chat turn: ZERO model calls after the stop, end to
    end through the real post-task lane. The loop's hard stop records the
    existing ``_skip_post_task_synthesis`` marker on the tool context;
    ``emit_task_results`` copies it onto the task before the root predicate
    runs, so the reflection worker is never dispatched and no open
    ``root_phase_checkpoint`` is seeded for the boot reconciler to re-pay. A
    positive control (the same turn, not stopped) proves the recording model
    would have seen the paid reflection call. Neither turn buys a narrative:
    both keep free host facts even when Stop prevents the worker from starting."""
    import ouroboros.llm as llm_mod
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN
    from tests.test_delivery_forced_finalization import _forced_test_context

    calls = []

    class RecordingLLM:
        def __init__(self, *_a, **_k):
            pass

        def chat(self, messages=None, model=None, **kw):
            calls.append({"model": model, "has_tools": bool(kw.get("tools"))})
            return ({"role": "assistant", "content": "recorded"},
                    {"cost": 0.0, "prompt_tokens": 1, "completion_tokens": 1})

    monkeypatch.setattr(llm_mod, "LLMClient", RecordingLLM)

    class InlineThread:  # the non-blocking lane, run inline so any paid call is recorded
        def __init__(self, *, target, daemon):
            assert daemon is True
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(pipeline.threading, "Thread", InlineThread)
    # The durable outbox (stamp + owed registration, no model call) must still cover the
    # stop notice: the marker skips PAID work only (the codex M3 finding).
    stamped, owed = [], []
    _stamp, _owe = pipeline.stamp_root_final_phase, pipeline.register_final_answer_owed
    monkeypatch.setattr(pipeline, "stamp_root_final_phase",
                        lambda send_event, task, **kw: stamped.append((task["id"], kw.get("post_task_open"))) or _stamp(send_event, task, **kw))
    monkeypatch.setattr(pipeline, "register_final_answer_owed",
                        lambda task, send_event, **kw: owed.append(task["id"]) or _owe(task, send_event, **kw))
    root = tmp_path / "data"
    (root / "logs").mkdir(parents=True)
    (root / "memory").mkdir()
    (root / "repo").mkdir()
    env = SimpleNamespace(drive_root=root, repo_dir=root / "repo", drive_path=lambda rel: root / rel)
    # Non-trivial and above the reflection threshold: both paid post-steps would fire.
    llm_trace = {"reasoning_notes": [], "tool_calls": [
        {"tool": "list_files", "arguments": {}, "result": "ok", "is_error": False}]}

    def _turn(task_id, *, stopped):
        loop, registry, ctx, _trace = _forced_test_context(root, usage={"rounds": 20, "cost": 0.02})
        registry._ctx.task_id = task_id
        registry._ctx.task_metadata = {"budget_drive_root": str(root), "root_task_id": task_id}
        if stopped:
            text, usage, _t = loop._handle_forced_finalization(ctx, REASON_OWNER_STOPPED_DIRECT_TURN)
            assert usage["reason_code"] == REASON_OWNER_REQUESTED_FINALIZATION
        else:
            text, usage = "Listed the root.", dict(ctx.accumulated_usage)
        task = {"id": task_id, "type": "task", "chat_id": 1, "_is_direct_chat": True,
                "text": "List the repository root and keep listing it until the owner stops you."}
        events = []
        calls.clear()
        pipeline.emit_task_results(
            env, object(), object(), events, task, text, usage, dict(llm_trace), 0.0,
            root / "logs", ctx=registry._ctx, event_queue=None,
        )
        return task, events, list(calls)

    task, events, stopped_calls = _turn("stopped1", stopped=True)
    assert stopped_calls == [], stopped_calls
    assert task.get("_skip_post_task_synthesis") is True
    assert [e["type"] for e in events] == ["send_message", "task_metrics", "task_done"]
    stored = pipeline.load_task_result(root, "stopped1") or {}
    assert stored.get("status") == "failed", stored
    assert "root_phase_checkpoint" not in stored, stored  # nothing for the boot reconciler to re-pay
    assert ("stopped1", False) in stamped and "stopped1" in owed   # outbox insurance kept, synthesis closed
    # The closed-phase stamp names the turn's ACTUAL terminal word: the durable row is
    # "failed" (owner_requested_finalization), and the stamp is what the chat row persists
    # and replay reads as the card's phase — a blanket "completed" would flip it to Done.
    from supervisor.terminal_delivery import pending_deliveries

    assert events[0]["progress_meta"]["task_terminal_status"] == "failed", events[0]
    owed_rows = [row for row in pending_deliveries(root) if row.get("task_id") == "stopped1"]
    assert owed_rows and owed_rows[0]["progress_meta"]["task_terminal_status"] == "failed", owed_rows

    def _summary_kinds(task_id):
        chat_log = root / "logs" / "chat.jsonl"
        rows = [json.loads(line) for line in chat_log.read_text(encoding="utf-8").splitlines()] if chat_log.exists() else []
        return [row.get("summary_kind") for row in rows if row.get("type") == "task_summary" and row.get("task_id") == task_id]

    stopped_kinds = _summary_kinds("stopped1")  # no paid post-task worker was dispatched
    assert stopped_kinds == ["host_task_facts"]  # free facts still reach pruned-result readers

    _task, _events, control_calls = _turn("control1", stopped=False)
    assert len(control_calls) >= 1, control_calls
    control_kinds = _summary_kinds("control1")
    assert control_kinds.count("host_task_facts") == 1 and "authored_root_summary" not in control_kinds


# --- "Stop now" while the paid synthesis is ALREADY in flight (audit point 4, G18) ---

_STAGES = ("chat_consolidation", "scratchpad_consolidation", "reflection", "promotion")


def _stubbed_stages(monkeypatch, calls, *, on_first=None):
    """Record the free facts row and the four paid stages in order; ``on_first`` runs
    INSIDE stage 1 (the Stop lands after the synthesis has begun, past the entry snapshot)."""
    import ouroboros.llm as llm_mod
    import ouroboros.post_task_evolution as pte

    monkeypatch.setattr(llm_mod, "LLMClient", lambda *a, **k: object())

    def _stage(name, ret=None):
        def _f(*a, **k):
            calls.append(name)
            if name == "chat_consolidation" and on_first is not None:
                on_first()
            return ret
        return _f

    monkeypatch.setattr(pipeline, "_run_chat_consolidation", _stage("chat_consolidation"))
    monkeypatch.setattr(pipeline, "_run_scratchpad_consolidation", _stage("scratchpad_consolidation"))
    monkeypatch.setattr(pipeline, "_record_task_facts", _stage("facts"))
    monkeypatch.setattr(pipeline, "_run_reflection", _stage(
        "reflection", {"reflection": "x", "backlog_candidates": [], "memory_actions": []}))
    monkeypatch.setattr(pipeline, "_update_improvement_backlog", _stage("promotion"))
    monkeypatch.setattr(pipeline, "_apply_reflection_memory_actions", lambda *a, **k: None)
    monkeypatch.setattr(pte, "maybe_promote", lambda *a, **k: None)


def _synthesis_root(tmp_path):
    root = tmp_path / "data"
    for rel in ("logs", "memory", "repo"):
        (root / rel).mkdir(parents=True)
    return root, SimpleNamespace(drive_root=root, repo_dir=root / "repo", drive_path=lambda rel: root / rel)


def _finalized_events(root, task_id):
    rows = [json.loads(line) for line in (root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if row.get("type") == "task_cost_finalized" and row.get("task_id") == task_id]


@pytest.mark.parametrize("stop", ["intent", "live_marker"])
def test_stop_now_during_inflight_synthesis_skips_the_remaining_paid_stages(tmp_path, monkeypatch, stop):
    """The Stop lands AFTER stage 1 began (the loop has returned, the entry
    snapshot saw no marker): the durable immediate cancel intent every stop
    ingress mints — or the live task marker re-read — trips the per-stage gate,
    so stages 2..4 never run, the checkpoint settles ``degraded`` and the typed
    ``post_task_stop_reason`` NAMES the skipped stages, riding the result row
    and the ``task_cost_finalized`` event alike. The in-flight key is gone."""
    from ouroboros.cancel_intents import STOP_POLICY_IMMEDIATE, request_cancel

    root, env = _synthesis_root(tmp_path)
    task_id = "inflight1"
    task = {"id": task_id, "type": "task", "chat_id": 1, "_is_direct_chat": True, "text": "keep listing"}
    pipeline._store_task_result(env, task, "Listed.", {"rounds": 20, "cost": 0.02}, {"tool_calls": [], "reasoning_notes": []})
    calls = []

    def _deliver_stop():
        if stop == "intent":
            request_cancel(root, task_id, source="http_single", allow_settled_target=True,
                           requested_stop_policy=STOP_POLICY_IMMEDIATE)
        else:
            task["_skip_post_task_synthesis"] = True   # the LIVE dict, not the entry snapshot

    _stubbed_stages(monkeypatch, calls, on_first=_deliver_stop)
    pipeline._run_post_task_processing_async(
        env, task, {"rounds": 20, "cost": 0.02}, {"tool_calls": [], "reasoning_notes": []}, {}, root / "logs", blocking=True)

    assert calls == ["facts", "chat_consolidation"], calls
    checkpoint = (pipeline.load_task_result(root, task_id) or {}).get("root_phase_checkpoint") or {}
    assert checkpoint.get("post_task_synthesis") == "degraded", checkpoint
    assert checkpoint.get("post_task_stop_reason") == (
        "owner_stopped:skipped=scratchpad_consolidation,reflection,promotion"), checkpoint
    finalized = _finalized_events(root, task_id)
    assert len(finalized) == 1 and finalized[0]["post_task_status"] == "degraded", finalized
    assert finalized[0]["post_task_stop_reason"] == checkpoint["post_task_stop_reason"], finalized
    assert (str(root.resolve()), task_id) not in pipeline._POST_TASK_SYNTHESIS_INFLIGHT


def test_no_stop_runs_every_paid_stage_and_finalizes_without_a_stop_reason(tmp_path, monkeypatch):
    """Positive control for the gate: an un-stopped synthesis records its free
    facts row, then runs all four paid stages in order, and the checkpoint is
    byte-identical to before (``completed``, no ``post_task_stop_reason`` anywhere)."""
    root, env = _synthesis_root(tmp_path)
    task_id = "unstopped1"
    task = {"id": task_id, "type": "task", "chat_id": 1, "_is_direct_chat": True, "text": "keep listing"}
    pipeline._store_task_result(env, task, "Listed.", {"rounds": 20, "cost": 0.02}, {"tool_calls": [], "reasoning_notes": []})
    calls = []
    _stubbed_stages(monkeypatch, calls)
    pipeline._run_post_task_processing_async(
        env, task, {"rounds": 20, "cost": 0.02}, {"tool_calls": [], "reasoning_notes": []}, {}, root / "logs", blocking=True)

    assert calls == ["facts", *_STAGES], calls
    checkpoint = (pipeline.load_task_result(root, task_id) or {}).get("root_phase_checkpoint") or {}
    assert checkpoint.get("post_task_synthesis") == "completed", checkpoint
    assert "post_task_stop_reason" not in checkpoint, checkpoint
    finalized = _finalized_events(root, task_id)
    assert len(finalized) == 1 and "post_task_stop_reason" not in finalized[0], finalized


def test_entry_marker_still_skips_every_paid_stage_and_seeds_no_checkpoint(tmp_path, monkeypatch):
    """The rc.14 path is unchanged: a Stop that landed inside the loop (marker
    on the task at entry) pays nothing and leaves no open checkpoint for the
    boot reconciler — the gate trips before stage 1 and the marker keeps the
    checkpoint writer's root predicate False. The free facts row still lands."""
    root, env = _synthesis_root(tmp_path)
    task_id = "marked1"
    task = {"id": task_id, "type": "task", "chat_id": 1, "_is_direct_chat": True,
            "text": "keep listing", "_skip_post_task_synthesis": True}
    pipeline._store_task_result(env, task, "Stopped.", {"rounds": 2, "cost": 0.01}, {"tool_calls": [], "reasoning_notes": []})
    calls = []
    _stubbed_stages(monkeypatch, calls)
    pipeline._run_post_task_processing_async(
        env, task, {"rounds": 2, "cost": 0.01}, {"tool_calls": [], "reasoning_notes": []}, {}, root / "logs", blocking=True)

    assert calls == ["facts"], calls
    stored = pipeline.load_task_result(root, task_id) or {}
    assert "root_phase_checkpoint" not in stored, stored
    assert not (root / "logs" / "events.jsonl").exists() or _finalized_events(root, task_id) == []


@pytest.mark.parametrize("role", ["root", "subagent"])
def test_requested_file_result_completes_without_committing_the_worktree(tmp_path, role):
    """An edited file is a valid requested result; dirty Git state adds no failure.

    Retains the contributor's isolated tracked-diff fixture, while asserting the
    owner-approved contract instead of universal commit-or-fail finalization.
    """
    import subprocess

    repo = tmp_path / "workspace"
    repo.mkdir()
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout

    git("init", "-q")
    (repo / "answer.txt").write_text("old\n", encoding="utf-8")
    git("add", "answer.txt")
    git("-c", "user.email=t@example.com", "-c", "user.name=T", "commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    (repo / "answer.txt").write_text("42\n", encoding="utf-8")
    task = {"id": "file-result", "type": "task", "repo_dir": str(repo), "delegation_role": role,
            "text": "Write 42 to answer.txt and leave the edited file for me.",
            "expected_output": "The edited answer.txt file; no Git commit requested."}
    pipeline._store_task_result(
        env=SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path / "host_tree"), task=task,
        text="answer.txt contains 42.", usage={"rounds": 1, "cost": 0},
        llm_trace={"tool_calls": [{"tool": "write_file", "status": "ok",
                                   "args": {"path": "answer.txt", "content": "42\n"}}]},
    )
    stored = pipeline.load_task_result(tmp_path, "file-result")
    assert stored["status"] == "completed"
    assert stored["reason_code"] != "work_uncommitted"
    assert stored["result"] == "answer.txt contains 42."
    assert (repo / "answer.txt").read_text(encoding="utf-8") == "42\n"
    assert git("rev-parse", "HEAD") == base
    assert "+42" in git("diff", "--", "answer.txt")


@pytest.mark.parametrize("artifact_status", ["failed", "missing"])
def test_terminal_event_reports_failed_bundle_over_older_capture_status(tmp_path, monkeypatch, artifact_status):
    monkeypatch.setattr(pipeline, "_store_task_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "load_task_result", lambda *args, **kwargs: {
        "status": "completed", "artifact_status": "ready", "artifact_bundle": {"status": artifact_status},
    })
    pending = []
    logs = tmp_path / "logs"
    logs.mkdir()
    pipeline.emit_task_results(
        env=SimpleNamespace(drive_root=tmp_path), memory=object(), llm=object(), pending_events=pending,
        task={"id": "capture", "type": "task", "chat_id": 1, "text": "deliver"}, text="Finished work",
        usage={"rounds": 1, "cost": 0.2}, llm_trace={"tool_calls": [], "reasoning_notes": []},
        start_time=0.0, drive_logs=logs, ctx=SimpleNamespace(pending_restart_reason=""),
    )
    terminal = next(item for item in pending if item["type"] == "task_done")
    assert terminal["artifact_status"] == artifact_status
    assert terminal["status"] == "completed"


def test_task_done_carries_the_custody_debt_list_the_row_holds(tmp_path, monkeypatch):
    """P5 S1: one debt rule on every surface, from one source.

    The owner-facing Reason line names the custody warning only while the row's
    own ``delegated_runs_unreconciled`` list is non-empty. The durable row
    carries that list; the live ``task_done`` event did not, so the card had to
    guess from the stamped code alone and the browser and the host rendered the
    same record differently. The event now copies the stored list whenever the
    row has one, and absent stays absent: a record with no list states nothing
    about the debt, on either surface.
    """
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *args, **kwargs: None)
    from ouroboros.task_results import write_task_result

    drive_logs = tmp_path / "logs"
    drive_logs.mkdir()
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    write_task_result(tmp_path, "custody-open", "running", delegated_runs_unreconciled=["run-a1"])

    owed_events = []
    pipeline.emit_task_results(
        env=env, memory=object(), llm=object(), pending_events=owed_events,
        task={"id": "custody-open", "root_task_id": "custody-open", "type": "task",
              "chat_id": 1, "text": "delegate"},
        text="done", usage={"rounds": 1, "cost": 0.0},
        llm_trace={"tool_calls": [], "reasoning_notes": []},
        start_time=0.0, drive_logs=drive_logs, ctx=SimpleNamespace(pending_restart_reason=""),
    )
    owed = next(row for row in owed_events if row["type"] == "task_done")
    stored = pipeline.load_task_result(tmp_path, "custody-open")
    assert owed["delegated_runs_unreconciled"] == ["run-a1"] == stored["delegated_runs_unreconciled"]
    assert owed["reason_code"] == "delegated_custody_unreconciled"

    plain_events = []
    pipeline.emit_task_results(
        env=env, memory=object(), llm=object(), pending_events=plain_events,
        task={"id": "custody-none", "root_task_id": "custody-none", "type": "task",
              "chat_id": 1, "text": "no delegation"},
        text="done", usage={"rounds": 1, "cost": 0.0},
        llm_trace={"tool_calls": [], "reasoning_notes": []},
        start_time=0.0, drive_logs=drive_logs, ctx=SimpleNamespace(pending_restart_reason=""),
    )
    plain = next(row for row in plain_events if row["type"] == "task_done")
    assert "delegated_runs_unreconciled" not in plain
