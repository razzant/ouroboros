"""Test that task_done event is NOT written directly to events.jsonl by emit_task_results.

The audit trail for task_done must go through EVENT_Q → supervisor _handle_task_done,
ensuring that send_message reaches the UI before task_done (causal ordering).
"""

import pathlib
import json


def test_task_done_dispatch_is_deferred_off_the_intake_loop(monkeypatch, tmp_path):
    """The task_done finalization (child ref-tree promotion + workspace patch)
    must not run on the supervisor intake loop: dispatch_event returns
    immediately while the handler is still blocked on the finalize pool."""
    import threading
    import time

    from supervisor import events as events_mod

    entered = threading.Event()
    release = threading.Event()

    def blocking_handler(evt, ctx):
        entered.set()
        release.wait(5.0)

    monkeypatch.setattr(events_mod, "_handle_task_done", blocking_handler)

    class Ctx:
        DRIVE_ROOT = tmp_path

        def append_jsonl(self, path, data):
            from ouroboros.utils import append_jsonl

            append_jsonl(path, data)

    t0 = time.monotonic()
    events_mod.dispatch_event(
        {"type": "task_done", "task_id": "t-defer", "status": "completed"},
        Ctx(),
    )
    elapsed = time.monotonic() - t0
    try:
        assert elapsed < 1.0, f"dispatch blocked the intake loop for {elapsed:.2f}s"
        assert entered.wait(2.0), "the deferred handler never started on the pool"
        # The handler is still blocked on `release`, yet dispatch returned.
    finally:
        release.set()


def test_deferred_task_done_failure_is_surfaced(monkeypatch, tmp_path):
    """A crashing deferred task_done handler must still produce the
    worker_event_handler_error record dispatch_event wrote for the sync path."""
    import threading
    import time

    from supervisor import events as events_mod

    def raising_handler(evt, ctx):
        raise RuntimeError("boom")

    monkeypatch.setattr(events_mod, "_handle_task_done", raising_handler)

    class Ctx:
        DRIVE_ROOT = tmp_path

        def append_jsonl(self, path, data):
            from ouroboros.utils import append_jsonl

            append_jsonl(path, data)

    events_mod.dispatch_event(
        {"type": "task_done", "task_id": "t-raise", "status": "completed"},
        Ctx(),
    )
    sup = tmp_path / "logs" / "supervisor.jsonl"
    deadline = time.time() + 5
    found = False
    while time.time() < deadline and not found:
        if sup.exists():
            for line in sup.read_text().splitlines():
                row = json.loads(line)
                if (
                    row.get("type") == "worker_event_handler_error"
                    and row.get("event_type") == "task_done"
                ):
                    found = True
                    break
        if not found:
            time.sleep(0.05)
    assert found, "the deferred handler's exception was not surfaced"


def test_task_done_submit_stamps_and_clears_finalization_pending(monkeypatch, tmp_path):
    """A completed task queued for finalization must be marked so the timeout
    enforcer does not reap it as stale while its RUNNING row keeps aging."""
    import threading
    import time

    from supervisor import events as events_mod
    from supervisor import queue as queue_mod

    queue_mod.RUNNING["t-custody"] = {
        "task": {"type": "task"},
        "started_at": time.time(),
        "last_progress_at": 0.0,
    }

    entered = threading.Event()
    release = threading.Event()

    def blocking_handler(evt, ctx):
        entered.set()
        release.wait(5.0)

    monkeypatch.setattr(events_mod, "_handle_task_done", blocking_handler)

    class Ctx:
        DRIVE_ROOT = tmp_path

        def append_jsonl(self, path, data):
            from ouroboros.utils import append_jsonl

            append_jsonl(path, data)

    try:
        events_mod.dispatch_event(
            {"type": "task_done", "task_id": "t-custody", "status": "completed"},
            Ctx(),
        )
        meta = queue_mod.RUNNING["t-custody"]
        assert meta.get("finalization_pending") is True
        assert meta["last_progress_at"] > 0.0
        assert entered.wait(2.0)
        release.set()
        deadline = time.time() + 5
        while time.time() < deadline:
            if not queue_mod.RUNNING["t-custody"].get("finalization_pending"):
                break
            time.sleep(0.05)
        assert not queue_mod.RUNNING["t-custody"].get("finalization_pending")
    finally:
        release.set()
        queue_mod.RUNNING.pop("t-custody", None)



def _make_fake_env(drive_root: pathlib.Path):
    """Create a minimal mock env for emit_task_results."""

    class FakeMemory:
        def load_identity(self):
            return "test identity"

    class FakeCtx:
        pending_restart_reason = None

    class FakeEnv:
        def __init__(self, root):
            self.drive_root = root

        def drive_path(self, sub):
            p = self.drive_root / sub
            p.mkdir(parents=True, exist_ok=True)
            return p

    return FakeEnv(drive_root), FakeMemory(), FakeCtx()


def _make_fake_llm():
    class FakeLLM:
        def chat(self, **kwargs):
            return {"content": "summary"}, {"cost": 0}
    return FakeLLM()


class TestTaskDoneOrdering:
    """Verify emit_task_results does not write task_done directly to events.jsonl."""

    def test_emit_task_results_does_not_write_task_done_to_events_jsonl(self, tmp_path):
        drive_root = tmp_path / "data"
        drive_root.mkdir()
        logs = drive_root / "logs"
        logs.mkdir()
        events_file = logs / "events.jsonl"
        (drive_root / "memory").mkdir()
        (drive_root / "task_results").mkdir()

        env, memory, ctx = _make_fake_env(drive_root)
        llm = _make_fake_llm()
        pending_events = []

        task = {"id": "test123", "type": "task", "chat_id": 1, "text": "hello"}
        usage = {"cost": 0.01, "rounds": 3, "prompt_tokens": 100, "completion_tokens": 50}
        llm_trace = {"tool_calls": [], "reasoning_notes": []}

        # Monkeypatch consolidation to no-op (avoid LLM calls)
        import ouroboros.agent_task_pipeline as atp
        orig_chat_consol = atp._run_chat_consolidation
        orig_scratchpad_consol = atp._run_scratchpad_consolidation
        orig_post_task = atp._run_post_task_processing_async
        atp._run_chat_consolidation = lambda *a, **kw: None
        atp._run_scratchpad_consolidation = lambda *a, **kw: None
        atp._run_post_task_processing_async = lambda *a, **kw: None

        try:
            import time
            atp.emit_task_results(
                env=env, memory=memory, llm=llm,
                pending_events=pending_events,
                task=task, text="Reply text",
                usage=usage, llm_trace=llm_trace,
                start_time=time.time() - 1.0,
                drive_logs=logs,
                ctx=ctx,
            )
        finally:
            atp._run_chat_consolidation = orig_chat_consol
            atp._run_scratchpad_consolidation = orig_scratchpad_consol
            atp._run_post_task_processing_async = orig_post_task

        # Check events.jsonl: should have task_eval but NOT task_done
        if events_file.exists():
            lines = [json.loads(line) for line in events_file.read_text().strip().split("\n") if line.strip()]
            event_types = [e["type"] for e in lines]
            assert "task_done" not in event_types, (
                "task_done should NOT be written to events.jsonl by emit_task_results; "
                "it must go through EVENT_Q → supervisor _handle_task_done"
            )
            # task_eval is still expected to be written directly
            assert "task_eval" in event_types

    def test_pending_events_ordering(self, tmp_path):
        """Verify send_message comes before task_done in pending_events."""
        drive_root = tmp_path / "data"
        drive_root.mkdir()
        logs = drive_root / "logs"
        logs.mkdir()
        (drive_root / "memory").mkdir()
        (drive_root / "task_results").mkdir()

        env, memory, ctx = _make_fake_env(drive_root)
        llm = _make_fake_llm()
        pending_events = []

        task = {"id": "order_test", "type": "task", "chat_id": 1, "text": "hi"}
        usage = {"cost": 0.02, "rounds": 1, "prompt_tokens": 50, "completion_tokens": 20}
        llm_trace = {"tool_calls": [], "reasoning_notes": []}

        import ouroboros.agent_task_pipeline as atp
        orig_chat_consol = atp._run_chat_consolidation
        orig_scratchpad_consol = atp._run_scratchpad_consolidation
        orig_post_task = atp._run_post_task_processing_async
        atp._run_chat_consolidation = lambda *a, **kw: None
        atp._run_scratchpad_consolidation = lambda *a, **kw: None
        atp._run_post_task_processing_async = lambda *a, **kw: None

        try:
            import time
            atp.emit_task_results(
                env=env, memory=memory, llm=llm,
                pending_events=pending_events,
                task=task, text="Order test reply",
                usage=usage, llm_trace=llm_trace,
                start_time=time.time() - 0.5,
                drive_logs=logs,
                ctx=ctx,
            )
        finally:
            atp._run_chat_consolidation = orig_chat_consol
            atp._run_scratchpad_consolidation = orig_scratchpad_consol
            atp._run_post_task_processing_async = orig_post_task

        event_types = [e["type"] for e in pending_events]
        send_idx = event_types.index("send_message")
        done_idx = event_types.index("task_done")
        assert send_idx < done_idx, (
            f"send_message (idx={send_idx}) must come before task_done (idx={done_idx}) "
            f"in pending_events to ensure causal ordering at the UI"
        )


class TestSupervisorTaskDoneAuditTrail:
    """Verify _handle_task_done writes to events.jsonl."""

    def test_handle_task_done_writes_events_jsonl(self, tmp_path):
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        events_file = logs_dir / "events.jsonl"

        # Minimal context mock
        class MockCtx:
            DRIVE_ROOT = tmp_path
            RUNNING = {"test_td": {"task": {"type": "task"}}}
            WORKERS = {}
            PENDING = []

            def persist_queue_snapshot(self, reason=""):
                pass

            class bridge:
                @staticmethod
                def push_log(data):
                    pass

            def sort_pending(self):
                pass

            def load_state(self):
                return {}

            def save_state(self, st):
                pass

            def append_jsonl(self, path, data):
                from ouroboros.utils import append_jsonl
                append_jsonl(path, data)

        from supervisor.events import _handle_task_done
        evt = {
            "type": "task_done",
            "task_id": "test_td",
            "task_type": "task",
            "cost_usd": 0.05,
            "total_rounds": 5,
            "prompt_tokens": 200,
            "completion_tokens": 80,
            "ts": "2026-04-02T12:00:00Z",
        }
        ctx = MockCtx()
        # A durable SETTLED result precedes every honest task_done (GR2-3: the
        # supervisor validates the event against the disk unconditionally — a
        # blank-status event over an absent row is a lifecycle fault, refused).
        from ouroboros.task_results import write_task_result as _write_result

        _write_result(tmp_path, "test_td", "completed", result="done")

        # Seed the physical-attempt authority.  The task_done transport's
        # compatibility cost field must not override ledger truth.
        from ouroboros.usage_accounting import (
            AttemptRequest,
            mark_dispatched,
            reserve_attempt,
            settle_attempt,
        )

        attempt = reserve_attempt(AttemptRequest(
            model="openai/gpt-5.2",
            provider="openai",
            max_budget_usd=0.10,
            global_limit_usd=1.0,
            drive_root=tmp_path,
            task_id="test_td",
            root_task_id="test_td",
            category="task",
            source="test",
        ))
        mark_dispatched(attempt)
        settle_attempt(
            attempt,
            {"prompt_tokens": 200, "completion_tokens": 80},
            cost_usd=0.05,
            cost_final=True,
        )

        _handle_task_done(evt, ctx)

        assert events_file.exists(), "events.jsonl should be created by _handle_task_done"
        lines = [json.loads(line) for line in events_file.read_text().strip().split("\n") if line.strip()]
        task_done_entries = [e for e in lines if e["type"] == "task_done"]
        assert len(task_done_entries) == 1
        entry = task_done_entries[0]
        assert entry["task_id"] == "test_td"
        assert entry["cost_usd"] == 0.05
        assert entry["total_rounds"] == 1
