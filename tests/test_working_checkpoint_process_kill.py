"""A real interpreter running the real loop is SIGKILLed mid-work (#1563, claims 2 and 4).

In-process fault tests raise exceptions, and exceptions still run ``finally``
blocks and deferred ACKs. Here the first generation dies with no cleanup at
all: inside an accepted batch (one call, or a sequential/parallel pair whose
first effect already happened), or between a checkpoint's temp write and its
replace. The next generation goes through the real crash restore from the last
queue snapshot, the owner's Resume and the real loop. No effect runs twice,
every call of the interrupted batch stays UNKNOWN, content mail is incorporated
exactly once (ACKed only after a saved state holds it, delivered again
otherwise) and a torn write leaves the previous checkpoint intact.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ouroboros import working_checkpoint as wc
from ouroboros.owner_mailbox import drain_owner_entries
from ouroboros.task_results import STATUS_RUNNING, write_task_result
from ouroboros.test_environment import isolated_environment
from supervisor.events_budget import HOLD_SAVED_WORK, budget_hold_fact
from tests._budget_pause_exact_helpers import _install_queue
from tests.working_checkpoint_kill_host import (
    BATCHES,
    FIRST_MAIL,
    SECOND_MAIL,
    TASK_ID,
    record_effect,
    recorded_effects,
)

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [pytest.mark.serial, pytest.mark.skipif(
    os.name != "posix", reason="This fixture qualifies a POSIX SIGKILL of the child's process group only")]


def _kill_at(base: Path, scenario: str) -> dict:
    """Run generation 1 in its own interpreter and SIGKILL it at its kill point."""
    log_path = base / f"gen1-{scenario}.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "tests.working_checkpoint_kill_host", str(base), scenario],
            cwd=ROOT, env=isolated_environment(base / "env", ROOT), stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        try:
            deadline, facts = time.monotonic() + 120, None
            while facts is None:
                assert proc.poll() is None, log_path.read_text()
                assert time.monotonic() < deadline, "generation 1 never reached its kill point"
                try:
                    facts = json.loads((base / "killpoint.json").read_text())
                except (FileNotFoundError, ValueError):
                    time.sleep(0.05)
            assert facts["pid"] == proc.pid and facts["point"] == scenario
            os.killpg(proc.pid, signal.SIGKILL)
            assert proc.wait(30) == -signal.SIGKILL
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(30)
    return facts


def _prepare(tmp_path, monkeypatch):
    drive = tmp_path / "drive"
    drive.mkdir()
    queue, state, workers = _install_queue(drive, monkeypatch)
    write_task_result(drive, TASK_ID, STATUS_RUNNING, chat_id=0, root_task_id=TASK_ID)
    workers.RUNNING[TASK_ID] = {"task": {"id": TASK_ID, "type": "task", "chat_id": 0, "_attempt": 1,
                                         "root_task_id": TASK_ID}, "worker_id": 0, "attempt": 1}
    queue.persist_queue_snapshot(reason="main_loop")  # the supervisor's last tick before the crash
    return drive, queue, state, workers


def _crash_restore_and_resume(queue, state, workers, monkeypatch) -> dict:
    """The next boot restores from the last snapshot (no teardown ran); only Resume releases it."""
    workers.RUNNING.clear()
    workers.PENDING.clear()
    queue.restore_pending_from_snapshot()
    [row] = [task for task in workers.PENDING if task["id"] == TASK_ID]
    hold = budget_hold_fact(row)
    assert hold and hold["reason"] == HOLD_SAVED_WORK, "a crash never continues by itself"
    assert row["_attempt"] == 2 and row["_working_recovery"]["from_attempt"] == 1
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    assert queue.resume_budget_paused_task(TASK_ID)["ok"]
    assert budget_hold_fact(row) is None
    return dict(row["_working_recovery"], _stop_cause=hold.get("stop_cause"))


def _second_generation(base: Path, drive: Path, handoff: dict, monkeypatch) -> tuple:
    from ouroboros import loop
    from tests.test_loop_transport_wait import _loop_kwargs
    from tests.test_working_checkpoint import _registry

    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", lambda *a, **k: pytest.fail("no provider call"))
    registry = _registry(drive, monkeypatch, TASK_ID, 2)
    registry._ctx.working_recovery = handoff

    def replayed(*_args, **_kwargs):
        record_effect(base, "replayed")
        pytest.fail("a recovered attempt never executes a saved call again")

    monkeypatch.setattr(registry, "execute_result", replayed, raising=False)
    turns = []

    def dispatch(call, disposition, *, candidate_predicate=None, **kwargs):
        turns.append([dict(row) for row in call.messages])
        return {"role": "assistant", "content": "Audit finished from the saved state", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    kwargs = _loop_kwargs(drive, registry, [])
    kwargs["task_id"] = TASK_ID
    answer, _usage, _trace = loop.run_llm_loop(**kwargs)
    return answer, turns


def _mentions(rows: list, text: str) -> int:
    return sum(str(row.get("content") or "").count(text) for row in rows)


def _unacked_beyond(drive: Path, attempt: int, held: list) -> list:
    """Owner-text ACKs are scoped to one physical attempt; a recovered attempt holds
    the restored ``seen`` ids instead, so only mail outside both is undelivered."""
    return [row["msg_id"] for row in drain_owner_entries(drive, TASK_ID, set(held), attempt)]


@pytest.mark.parametrize("scenario", ["inflight_effect", "inflight_sequential_batch", "inflight_parallel_batch"])
def test_sigkill_inside_an_accepted_batch_never_replays_it_and_keeps_acked_mail_once(
        tmp_path, monkeypatch, scenario):
    _tool, call_ids = BATCHES[scenario]
    drive, queue, state, workers = _prepare(tmp_path, monkeypatch)
    facts = _kill_at(tmp_path, scenario)
    assert sorted(recorded_effects(tmp_path)) == sorted(call_ids), "every effect of the batch happened once"

    # What the dead interpreter left: the pre-effect state naming the whole batch
    # (no result of it was saved), holding the owner mail ACKed only after that save.
    saved = json.loads(wc.checkpoint_path(drive, TASK_ID, 1).read_bytes())
    assert saved["working"]["boundary"] == "pre_effect"
    assert saved["working"]["pending_tool_call_ids"] == call_ids
    assert "mail-1" in saved["seen"]
    assert _mentions(saved["messages"], FIRST_MAIL) == 1
    assert drain_owner_entries(drive, TASK_ID, set(), 1) == [], "saved content mail was acknowledged"

    handoff = _crash_restore_and_resume(queue, state, workers, monkeypatch)
    assert handoff["boundary"] == "pre_effect"
    answer, turns = _second_generation(tmp_path, drive, handoff, monkeypatch)

    assert answer == "Audit finished from the saved state"
    assert sorted(recorded_effects(tmp_path)) == sorted(call_ids), "no effect ran again"
    [first] = turns[:1]
    closures = {row.get("tool_call_id"): str(row.get("content")) for row in first if row.get("role") == "tool"}
    for call_id in call_ids:  # a finished call's result was never saved: it is as UNKNOWN as the live one
        assert "UNKNOWN" in closures[call_id] and "NOT re-executed" in closures[call_id]
    assert not any(_mentions(first, f"{call_id} completed") for call_id in call_ids), "no unsaved result appears"
    assert _mentions(first, FIRST_MAIL) == 1, "incorporated mail is neither lost nor duplicated"
    assert _unacked_beyond(drive, 2, saved["seen"]) == []
    assert not wc.checkpoint_path(drive, TASK_ID, 1).exists(), "the consumer retired the captured file"
    (tmp_path / "process-kill-facts.json").write_text(json.dumps(
        {"scenario": scenario, "killpoint": facts, "handoff": handoff, "first_turn_messages": len(first),
         "unknown_closures": {call_id: closures[call_id] for call_id in call_ids}}, indent=2))


def test_sigkill_between_checkpoint_temp_and_replace_keeps_previous_state_and_redelivers_mail(
        tmp_path, monkeypatch):
    drive, queue, state, workers = _prepare(tmp_path, monkeypatch)
    facts = _kill_at(tmp_path, "torn_ready_write")
    assert recorded_effects(tmp_path) == ["effect-1"]

    # The replace never happened: the file is the complete previous state, and
    # the half-written temp is an orphan beside it.
    path = wc.checkpoint_path(drive, TASK_ID, 1)
    saved = json.loads(path.read_bytes())
    assert saved["working"]["seq"] == facts["seq"] - 1 and saved["working"]["boundary"] == "post_batch"
    assert saved["working"]["pending_tool_call_ids"] == []
    assert "mail-1" in saved["seen"] and "mail-2" not in saved["seen"]
    temp = path.parent / facts["temp"]
    assert temp.is_file() and 0 < temp.stat().st_size < facts["full_bytes"]
    assert [row["msg_id"] for row in drain_owner_entries(drive, TASK_ID, set(), 1)] == ["mail-2"], \
        "content drained into an unsaved state was never acknowledged"

    handoff = _crash_restore_and_resume(queue, state, workers, monkeypatch)
    assert handoff["boundary"] == "post_batch" and handoff["seq"] == facts["seq"] - 1
    answer, turns = _second_generation(tmp_path, drive, handoff, monkeypatch)

    assert answer == "Audit finished from the saved state"
    assert recorded_effects(tmp_path) == ["effect-1"]
    [first] = turns[:1]
    assert _mentions(first, "receipt R-17 recorded") == 1, "the completed result is restored once"
    assert _mentions(first, FIRST_MAIL) == 1
    assert _mentions(first, SECOND_MAIL) == 1, "the unsaved mail is delivered again, once"
    assert _unacked_beyond(drive, 2, saved["seen"]) == [], "attempt 2 acknowledged the mail it incorporated"
    # Terminal discard owns only the named attempt files; the orphaned temp is the
    # existing stale-temp sweep's (every boot, once older than its age guard).
    wc.discard(drive, TASK_ID)
    assert temp.is_file()
    from ouroboros.utils import sweep_stale_temp_files

    assert sweep_stale_temp_files(drive, min_age_sec=3600.0, scripts=False) == 0, "a fresh temp is never reaped"
    assert sweep_stale_temp_files(drive, min_age_sec=0.0, scripts=False) == 1 and not temp.exists()
    (tmp_path / "process-kill-facts.json").write_text(json.dumps(
        {"scenario": "torn_ready_write", "killpoint": facts, "handoff": handoff,
         "orphan_temp_bytes": facts["full_bytes"] // 2}, indent=2))
