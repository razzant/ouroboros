"""A deep self-review under the configured silence window (owner 30 Oct, A; whole review R3).

A ``deep_self_review`` task runs ``review_change(subject=system)`` outside the loop and review custody,
so no round, usage row or tool lease reaches the supervisor until its delivery ends. The delivery holds
the review's own typed lease instead of a hidden one-hour idle floor: the tests below drive the real
worker branch into the real supervisor halves (its event dispatch table and its timeout pass), on a
clock the test moves, and never write an exemption into the supervisor's row themselves.
"""

from __future__ import annotations

import queue
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from ouroboros import deep_self_review
from ouroboros.deep_self_review import run_deep_self_review
from tests.test_deep_review_slot import _DEEP_SLOT_ID, _REPORT, _FakeSessionExecutor, _native_row, _ScriptedLLM

pytestmark = pytest.mark.usefixtures("provider_catalog_offline")


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    from ouroboros.tools import review_helpers

    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    (root / "BIBLE.md").write_text("# BIBLE\n\n## Principle 0: Agency\n\nOuroboros is a becoming personality.\n",
                                   encoding="utf-8")
    for name, text in (("ARCHITECTURE", "# Arch\n\n## Review stack\n\ntext\n"), ("DEVELOPMENT", "# Dev\n\n## Rules\n\nx\n"),
                       ("CHECKLISTS", "# Checks\n\n## Change Review Checklist\n\ny\n")):
        (root / "docs" / f"{name}.md").write_text(text, encoding="utf-8")
    monkeypatch.setattr(review_helpers, "REPO_ROOT", root)
    return root


@pytest.fixture()
def drive(tmp_path):
    root = tmp_path / "drive"
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "identity.md").write_text("I am Ouroboros.\n", encoding="utf-8")
    (root / "logs").mkdir()
    return root


class _Supervisor:
    """The supervisor's real halves over ONE running ``deep_self_review`` row, on a clock the test moves:
    the event dispatch table for the worker's ``cognitive_operation`` facts and the timeout pass at the
    configured 900 s silence window. No lifetime: only leases, progress and a deadline decide."""

    def __init__(self, monkeypatch, tmp_path, task, start):
        from supervisor import cognitive_operations
        from supervisor import queue as squeue

        self.now, self.events, self.facts = start, queue.Queue(), []
        self.meta = {"task": dict(task), "attempt": 1, "started_at": start, "last_progress_at": start, "worker_id": 0}
        self.running = {task["id"]: self.meta}
        self.ctx = SimpleNamespace(RUNNING=self.running, DRIVE_ROOT=tmp_path, append_jsonl=lambda *_a, **_k: None)
        monkeypatch.setattr(cognitive_operations, "time", SimpleNamespace(time=lambda: self.now))
        for name, value in (("DRIVE_ROOT", tmp_path), ("RUNNING", self.running), ("PENDING", []),
                            ("FINALIZATION_GRACE_SEC", 300), ("get_task_idle_timeout_sec", lambda: 900),
                            ("get_task_abs_ceiling_sec", lambda: None)):
            monkeypatch.setattr(squeue, name, value)

    def pump(self, at):
        """Every fact the worker emitted so far reaches the real dispatch table at ``at``."""
        from supervisor.events import dispatch_event

        self.now = at
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                return
            if event.get("type") == "cognitive_operation":
                self.facts.append((event["phase"], event["kind"], event["operation_id"]))
                dispatch_event(event, self.ctx)

    def silent_at(self, at):
        """One timeout pass at ``at``: whether it dated the row's silence."""
        from supervisor import queue as squeue

        self.now = at
        squeue._enforce_task_timeouts_locked(SimpleNamespace(WORKERS={}), at, 7, {})
        return "silence_attention" in self.meta


def test_a_deep_review_task_holds_its_own_lease_through_the_real_worker_branch(repo, drive, tmp_path, monkeypatch):
    """The real worker branch (agent → run_system_review → run_deep_self_review → the session executor) holds
    the review's lease, so 2000 s into a delivery with no other fact, past the 900 s window, the real timeout
    pass dates no silence; the close is a fresh progress stamp, so the tick right after it dates none either;
    then the window applies again: 901 s later the silence gets attention, never a stop."""
    import ouroboros.agent as agent_module
    import ouroboros.review_execution as review_execution
    from ouroboros.agent import Env, OuroborosAgent
    from ouroboros.config import IDLE_LEASE_MAX_SEC
    from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool

    set_review_pool(monkeypatch, pool_roster(pool_seat("selected-session", "codex=gpt-5.6-sol", kind="agent_session")))
    monkeypatch.setenv("OUROBOROS_TASK_ABS_CEILING_SEC", "unlimited")
    monkeypatch.delenv("OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC", raising=False)
    monkeypatch.setattr(deep_self_review, "_session_route_reason", lambda row: "")
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    monkeypatch.setattr(agent_module, "build_llm_messages", lambda **_k: ([], {}))
    answers = []
    monkeypatch.setattr(agent_module, "emit_task_results", lambda *a, **_k: answers.append(a[5]))
    task = {"id": "dsr-lease", "type": "deep_self_review", "chat_id": 1, "text": "owner:/review",
            "reviewer": "selected-session", "_attempt": 1}
    start = float(int(time.time()))
    supervisor = _Supervisor(monkeypatch, tmp_path, task, start)
    seen = {}

    class _LongDelivery(_FakeSessionExecutor):
        def execute(self):
            supervisor.pump(start)
            seen["leases"] = dict(supervisor.meta.get("active_operation_leases") or {})
            seen["silent"] = supervisor.silent_at(start + 2000.0)
            return super().execute()

    _FakeSessionExecutor.instances = []
    monkeypatch.setattr(review_execution, "_review_route_executor", _LongDelivery)
    OuroborosAgent(Env(repo_dir=repo, drive_root=drive), event_queue=supervisor.events).handle_task(task)

    assert answers and answers[-1].split("\n\nReview record: ", 1)[0].endswith(_REPORT)
    assignment = _FakeSessionExecutor.instances[-1].assignment
    assert assignment.slot.timeout_sec is None  # no window of its own: the lease is the supervisor's finite one
    assert seen["leases"] == {assignment.call_id: {
        "kind": "review", "until_ts": start + IDLE_LEASE_MAX_SEC, "task_attempt": 1, "execution_id": "",
        "round_id": "", "slot_id": assignment.slot.slot_id, "owner_id": "", "wait_for_tasks": []}}
    assert seen["silent"] is False
    supervisor.pump(start + 2000.0)  # the delivery ended: its close reaches the supervisor
    assert supervisor.facts == [("started", "review", assignment.call_id), ("finished", "review", assignment.call_id)]
    assert "active_operation_leases" not in supervisor.meta and supervisor.meta["last_progress_at"] == start + 2000.0
    assert supervisor.silent_at(start + 2000.5) is False
    assert supervisor.silent_at(start + 2000.0 + 901.0) is True
    assert supervisor.meta["silence_attention"]["last_progress_at"] == start + 2000.0
    assert "finalization_requested_at" not in supervisor.meta


class _Cancelled(BaseException):
    """An exit no ``except Exception`` sees, as a cancel unwinding the worker."""


@pytest.mark.parametrize("ending", ["answer", "error", "budget", "cancel"])
def test_the_native_delivery_lease_follows_the_owner_deadline_and_closes_on_every_exit(
        repo, drive, tmp_path, monkeypatch, ending):
    """The api row's native episode holds the same lease, bounded by the owner deadline (its window), and every
    exit closes it into a fresh progress stamp: an answer (``finished``); an error, a budget refusal or a
    cancel (``failed``), the last two still propagating."""
    from ouroboros.usage_accounting import BudgetExceeded

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OUROBOROS_TASK_ABS_CEILING_SEC", "unlimited")
    start = float(int(time.time()))
    deadline = datetime.fromtimestamp(start + 3000.0, timezone.utc).isoformat()
    task = {"id": "dsr-native", "type": "deep_self_review", "chat_id": 1, "deadline_at": deadline}
    supervisor = _Supervisor(monkeypatch, tmp_path, task, start)
    seen = {}

    class _Episode(_ScriptedLLM):
        def chat(self, **kwargs):
            supervisor.pump(start)
            seen["leases"] = dict(supervisor.meta.get("active_operation_leases") or {})
            seen["silent"] = supervisor.silent_at(start + 2000.0)
            failure = {"error": RuntimeError("socket reset"), "budget": BudgetExceeded("root budget exhausted"),
                       "cancel": _Cancelled()}.get(ending)
            if failure is not None:
                raise failure
            return super().chat(**kwargs)

    def run():
        return run_deep_self_review(repo, drive, _Episode([{"content": _REPORT}]), lambda _m: None, task_id=task["id"],
                                    deadline_at=deadline, slot=_native_row(),
                                    usage_ctx=SimpleNamespace(event_queue=supervisor.events, task_attempt=1))

    if ending in {"budget", "cancel"}:
        with pytest.raises(BudgetExceeded if ending == "budget" else _Cancelled):
            run()
    else:
        text, usage = run()
        assert text.endswith(_REPORT) if ending == "answer" else usage["reason_code"] == "deep_self_review_error"

    [(operation_id, lease)] = seen["leases"].items()
    assert (lease["kind"], lease["until_ts"], lease["slot_id"]) == ("review", start + 3000.0, _DEEP_SLOT_ID)
    assert seen["silent"] is False
    supervisor.pump(start + 2000.0)
    closed = "finished" if ending == "answer" else "failed"
    assert supervisor.facts == [("started", "review", operation_id), (closed, "review", operation_id)]
    assert "active_operation_leases" not in supervisor.meta and supervisor.meta["last_progress_at"] == start + 2000.0
    assert supervisor.silent_at(start + 2000.5) is False and supervisor.silent_at(start + 2901.0) is True
