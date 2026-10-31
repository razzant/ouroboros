"""#1562: on Resume the task hears what became of the delegated runs its owner Pause stopped.

The Pause stops the task's own runs (owner 2026-10-07 fork 1 = A); once a stop is confirmed
the run settles and leaves the liveness census, which is why that census alone cannot answer
the resumed task. The Pause keeps its participants on its own pause row or warm wait, under
that record's existing compare-and-set, and every Resume path (cold, warm, warm turned cold by
a Restart) reads each one's CURRENT custody and locally retained terminal source: no engine
call, no paid round, no successor. Driven through the real park, settle tick, Resume grant,
assignment and resumed loop, with a fake daemon only where the engine itself would answer.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _fast_hold, _install_queue, _loop_ctx
from tests.worker_pool_support import affirm_ready

RESUMABLE = {"cause": "cancelled", "carriers": ["session", "packet"]}
CONTINUITY = [{"tryIndex": 1, "attemptId": "att-1", "carrier": "session", "cause": "transport_fault",
               "from": {"profileId": "p1"}, "to": {"profileId": "p1"}, "memory": "kept",
               "observedModel": "m"}]


def _daemon(monkeypatch, remote):
    """The engine as the stop and its read-back see it; ``remote["up"]`` False is an unreachable gateway."""
    from ouroboros.gateways import claudexor as gw

    class Daemon:
        def handshake(self, **_kw):
            if not remote.get("up", True):
                raise RuntimeError("gateway down")
            return {"compatible": True}

        def cancel_run(self, run_id, reason=""):
            remote.setdefault("cancels", []).append(run_id)
            return {"accepted": True, "status": "accepted"}

        def get_run(self, run_id, **_kw):
            return {"lastSeq": 3, "summary": {"state": remote["state"], "spendUsd": 0.0,
                                              "outcomeFacts": {"reason": "host_cancelled"},
                                              "resumable": RESUMABLE, "continuity": CONTINUITY}}

        def close(self):
            return None

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: Daemon())


def _engine_offline(monkeypatch):
    from ouroboros.gateways import claudexor as gw

    def refuse(*_a, **_k):
        raise AssertionError("a Resume disclosure never calls the engine")

    monkeypatch.setattr(gw, "ClaudexorGateway", refuse)


def _own_run(root, task_id, run_id="run-own", *, invocation="inv-own"):
    from ouroboros import delegate_custody as dc

    assert dc.record_started(root, dc.RunCustody(
        run_id=run_id, task_id=task_id, route_id="codex", model="m", project_id="p", project_owned=False,
        root_task_id=task_id, ledger_root=str(root), invocation_id=invocation))


def _sup(tmp_path, queue, workers):
    return SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                           WORKERS=workers.WORKERS, sort_pending=lambda: None,
                           persist_queue_snapshot=queue.persist_queue_snapshot, bridge=None)


def _cold_park(tmp_path, monkeypatch, queue, workers, task_id="solo"):
    from ouroboros import budget_pause
    from supervisor.events_budget import install_exact_budget_pause

    ctx, limit = _loop_ctx(tmp_path, task_id)
    ctx.root_task_id = task_id
    _fast_hold(monkeypatch, budget_pause)
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.enter_owner_pause(limit)
    budget_pause.end_dispatch_fence(task_id)
    install_exact_budget_pause(_sup(tmp_path, queue, workers), task_id,
                               budget_pause.exact_pause_marker(raised.value.pause)["checkpoint"])
    return budget_pause.budget_pause_row(tmp_path, task_id)


def _resume_messages(tmp_path, monkeypatch, queue, workers, task_id="solo"):
    """The owner's Resume through grant, assignment and the resumed loop: the model's first facts."""
    from ouroboros import budget_pause

    granted = queue.resume_budget_paused_task(task_id)
    assert granted["ok"] is True, granted
    sent = []
    workers.WORKERS[0] = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                                                      in_q=SimpleNamespace(put=lambda task: sent.append(dict(task)))))
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [task_id]
    ctx, limit = _loop_ctx(tmp_path, task_id)
    ctx.budget_pause_resume = sent[0]["_budget_pause_resume"]
    state = budget_pause.load_budget_pause(ctx)
    monkeypatch.setattr("ouroboros.owner_wait.rebind_restored_route", lambda *_a, **_k: (None, "max"))
    monkeypatch.setattr("ouroboros.owner_wait.restore_continuation_state", lambda *_a, **_k: None)
    messages = list(limit.messages)
    budget_pause.resume_paused_loop(limit.tools, state, messages, {}, {}, set(), budget_remaining_usd=5.0)
    return [str(message.get("content") or "") for message in messages if message.get("role") == "user"]


@pytest.mark.parametrize("gateway_at_park", ["up", "down"], ids=["stop-requested", "stop-unknown-then-known"])
def test_a_cold_resume_names_what_became_of_the_runs_the_pause_stopped(tmp_path, monkeypatch, gateway_at_park):
    """The run the Pause stopped settled before Resume, so the census is empty; the resumed
    task still hears its stop outcome beside the engine's own terminal, retained continuation
    facts and output, and the exact inspect/continue verbs, from local custody alone."""
    from ouroboros import budget_pause, owner_pause
    from ouroboros.delegate_custody_current import task_snapshot
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from supervisor import owner_pause_control
    from supervisor.owner_pause_control import request_owner_pause

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    write_task_result(tmp_path, "solo", STATUS_RUNNING, chat_id=0)
    workers.RUNNING["solo"] = {"task": {"id": "solo", "type": "task", "chat_id": 0, "root_task_id": "solo"},
                               "worker_id": 0, "attempt": 1}
    _own_run(tmp_path, "solo")
    remote = {"state": "running", "up": gateway_at_park == "up"}
    _daemon(monkeypatch, remote)
    assert request_owner_pause("solo", request_id="press")["ok"]

    parked = _cold_park(tmp_path, monkeypatch, queue, workers)
    assert parked["settlement"] == owner_pause.SETTLEMENT_EXTERNAL_RUNNING
    first_outcome = "requested" if gateway_at_park == "up" else "not_issued_gateway_unavailable"
    assert [(row["run_id"], row["stop_requested"], row["stop_outcome"]) for row in parked["episode_runs"]] == [
        ("run-own", True, first_outcome)]

    # The run obeys the stop; only the settle tick notices. The census then names nothing.
    remote.update(state="cancelled", up=True)
    monkeypatch.setattr(owner_pause_control, "_LAST_SETTLE_CHECK", {})
    assert owner_pause_control.settle_requested_owner_pauses(queue, now=100.0) == ["solo"]
    settled = budget_pause.budget_pause_row(tmp_path, "solo")
    assert [run["state"] for run in settled["external_runs"]["runs"]] == ["stop_confirmed"]
    assert [(row["run_id"], row["stop_outcome"]) for row in settled["episode_runs"]] == [("run-own", "confirmed")]
    assert task_snapshot(tmp_path, "solo")["runs"]["run-own"].settled is True

    _engine_offline(monkeypatch)
    notices = _resume_messages(tmp_path, monkeypatch, queue, workers)
    resumed = budget_pause.budget_pause_row(tmp_path, "solo")
    assert resumed["external_runs"]["runs"] == [], "the grant's fresh census: a settled run never blocks again"
    assert resumed["episode_runs"] == settled["episode_runs"]
    notice = next(text for text in notices if "continued from its owner Pause" in text)
    line = next(text for text in notice.splitlines() if text.startswith("- run run-own"))
    assert "(codex): this Pause requested its stop (stop outcome: confirmed)" in line
    assert "engine terminal: cancelled (host_cancelled)" in line
    assert "engine resumable facts: " + json.dumps(RESUMABLE, sort_keys=True) in line
    assert "try 1 (att-1): session after transport_fault" in line
    assert 'it may be continued: delegate_start(subagent_id=..., continue_from="run-own")' in line
    assert 'inspect: delegate_wait(run_id="run-own")' in line
    assert "never start a second writer over such a run" in notice
    # Facts only: nothing was started, continued or granted by the disclosure itself.
    assert set(task_snapshot(tmp_path, "solo")["runs"]) == {"run-own"}
    assert [task["id"] for task in workers.PENDING] == [] and remote["cancels"] == (
        ["run-own"] if gateway_at_park == "up" else ["run-own"])


def test_a_new_pause_of_the_same_task_never_inherits_the_last_ones_participants(tmp_path, monkeypatch):
    """Cross-episode: the next Pause records only what ITS observations named."""
    from ouroboros import budget_pause
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from supervisor import owner_pause_control
    from supervisor.owner_pause_control import request_owner_pause

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    write_task_result(tmp_path, "solo", STATUS_RUNNING, chat_id=0)
    workers.RUNNING["solo"] = {"task": {"id": "solo", "type": "task", "chat_id": 0, "root_task_id": "solo"},
                               "worker_id": 0, "attempt": 1}
    _own_run(tmp_path, "solo")
    remote = {"state": "running"}
    _daemon(monkeypatch, remote)
    assert request_owner_pause("solo", request_id="first")["ok"]
    first = _cold_park(tmp_path, monkeypatch, queue, workers)
    remote["state"] = "cancelled"
    monkeypatch.setattr(owner_pause_control, "_LAST_SETTLE_CHECK", {})
    owner_pause_control.settle_requested_owner_pauses(queue, now=100.0)
    _resume_messages(tmp_path, monkeypatch, queue, workers)

    assert request_owner_pause("solo", request_id="second")["ok"]
    ctx, limit = _loop_ctx(tmp_path, "solo")
    ctx.root_task_id = "solo"
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.enter_owner_pause(limit)
    budget_pause.end_dispatch_fence("solo")
    second = raised.value.pause
    assert second["pause_id"] != first["pause_id"]
    assert second["episode_runs"] == [], "the settled run of the first Pause is not this Pause's participant"
    from ouroboros.external_runs import paused_runs_notice

    assert paused_runs_notice(ctx, second) == "- none named"


def _warm_author(tmp_path, monkeypatch, remote):
    """A direct root the owner Paused while its own run still ran: the actor parks warm."""
    from ouroboros import owner_wait
    from ouroboros.model_wait import TaskModelWait
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from supervisor.owner_pause_control import request_owner_pause
    from tests.test_direct_chat_turn_owner_control import _live_chat_agent

    queue, state, _workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    write_task_result(tmp_path, "author", STATUS_RUNNING, chat_id=0)
    _live_chat_agent(monkeypatch, task_id="author")
    _own_run(tmp_path, "author")
    _daemon(monkeypatch, remote)
    assert request_owner_pause("author", request_id="press")["state"] == "requested"
    ctx, limit = _loop_ctx(tmp_path, "author", direct=True)
    ctx.pending_events, ctx.current_chat_id = [], 0
    waiter = TaskModelWait(task={"id": "author", "budget_drive_root": str(tmp_path)},
                           drive_root=tmp_path, event_queue=None, worker_slot_held=False)
    waiter.tool_context, ctx.model_wait_context = ctx, waiter
    ctx.owner_wait_callback = owner_wait.direct_owner_wait
    return queue, ctx, limit


def _park_warm_then(tmp_path, monkeypatch, remote, queue, ctx, limit, *, resume, before_tick=None):
    """Park warm; while parked the run obeys and the settle tick records it; then Resume or stop here."""
    import time

    from ouroboros import owner_pause, owner_wait
    from ouroboros.task_results import load_task_result
    from supervisor import owner_pause_control
    from supervisor.budget_resume import resume_warm_owner_pause_root

    real_sleep, box = time.sleep, {}

    def poll(seconds):  # the parked direct actor's poll
        if "tick" in box:
            return real_sleep(min(seconds, 0.01))
        box["parked"] = load_task_result(tmp_path, "author")["owner_wait"]
        if before_tick is not None:
            before_tick(box["parked"])
        remote["state"] = "cancelled"
        box["tick"] = owner_pause_control.settle_requested_owner_pauses(queue, now=100.0)
        box["recorded"] = load_task_result(tmp_path, "author")["owner_wait"]
        if resume:
            box["resume"] = resume_warm_owner_pause_root("author")
            return None
        raise _Stopped()

    monkeypatch.setattr(owner_pause_control, "_LAST_SETTLE_CHECK", {})
    monkeypatch.setattr(owner_wait.time, "sleep", poll)
    try:
        box["cause"] = owner_wait.park_owner_pause_warm(limit, ctx, fence=owner_pause.read_fence(tmp_path, "author"),
                                                        detached=[])
    except _Stopped:
        pass
    return box


class _Stopped(Exception):
    """The application stops while the park is warm (the test ends the actor there)."""


def test_a_warm_resume_names_the_runs_its_park_stopped_after_they_left_the_census(tmp_path, monkeypatch):
    remote = {"state": "running"}
    queue, ctx, limit = _warm_author(tmp_path, monkeypatch, remote)
    box = _park_warm_then(tmp_path, monkeypatch, remote, queue, ctx, limit, resume=True)
    assert box["resume"]["ok"] and box["cause"] == "control:owner_resume", box
    parked, recorded = box["parked"]["owner_pause"], box["recorded"]["owner_pause"]
    assert [(row["run_id"], row["stop_outcome"]) for row in parked["episode_runs"]] == [("run-own", "requested")]
    assert [run["state"] for run in recorded["external_runs"]["runs"]] == ["stop_confirmed"]
    assert [(row["run_id"], row["stop_outcome"]) for row in recorded["episode_runs"]] == [("run-own", "confirmed")]
    notice = next(str(message["content"]) for message in reversed(limit.messages)
                  if "has now resumed it" in str(message.get("content") or ""))
    line = next(text for text in notice.splitlines() if text.startswith("- run run-own"))
    assert "this Pause requested its stop (stop outcome: confirmed)" in line
    assert "engine terminal: cancelled (host_cancelled)" in line and 'continue_from="run-own"' in line


def test_a_warm_park_turned_cold_by_restart_keeps_its_participants_for_the_cold_resume(tmp_path, monkeypatch):
    """warm -> Restart -> cold: the conversion carries the park's participants onto the exact
    cold pause, and the cold Resume's reader names them from current custody."""
    from ouroboros.external_runs import paused_runs_notice
    from supervisor.restart_retention import retain_owner_pause_park

    remote = {"state": "running"}
    queue, ctx, limit = _warm_author(tmp_path, monkeypatch, remote)
    box = _park_warm_then(tmp_path, monkeypatch, remote, queue, ctx, limit, resume=False)
    assert "cause" not in box and box["recorded"]["state"] == "waiting"
    cold = retain_owner_pause_park(tmp_path, "author", 1)
    assert cold["reason"] == "owner" and cold["external_runs"] == box["recorded"]["owner_pause"]["external_runs"]
    assert cold["episode_runs"] == box["recorded"]["owner_pause"]["episode_runs"]
    _engine_offline(monkeypatch)
    lines = paused_runs_notice(ctx, cold).splitlines()
    assert len(lines) == 1 and lines[0].startswith("- run run-own (codex): this Pause requested its stop")
    assert "engine terminal: cancelled (host_cancelled)" in lines[0]


# --- the participant list and its reader, case by case ---------------------------------------

def test_participants_merge_a_late_bound_start_and_never_lose_a_stop_or_a_row():
    from ouroboros.external_runs import episode_runs

    park = {"runs": [
        {"run_id": "", "invocation_id": "inv-b", "route": "claude", "stop_policy": "reconcile_first",
         "state": "stop_unknown", "stop_outcome": "pending_invocation_unbound", "review_owned": False},
        {"run_id": "run-a", "invocation_id": "inv-a", "route": "codex", "stop_policy": "request_stop",
         "state": "stop_unknown", "stop_outcome": "not_issued_gateway_unavailable", "review_owned": False},
        {"run_id": "run-r", "invocation_id": "inv-r", "route": "codex", "stop_policy": "observe_only",
         "state": "running", "stop_outcome": "", "review_owned": True}]}
    rows = episode_runs([], park)
    tick = {"runs": [
        {"run_id": "run-b", "invocation_id": "inv-b", "stop_policy": "request_stop", "stop_outcome": "requested"},
        {"run_id": "run-a", "invocation_id": "inv-a", "stop_policy": "request_stop", "stop_outcome": "confirmed"}]}
    rows = episode_runs(rows, tick)
    rows = episode_runs(rows, {"runs": [{"run_id": "run-a", "stop_policy": "observe_only", "stop_outcome": ""}]})
    rows = episode_runs(rows, {"runs": []})
    assert [(row.get("run_id"), row.get("invocation_id"), row.get("stop_requested", False), row.get("stop_outcome", ""))
            for row in rows] == [("run-b", "inv-b", True, "requested"), ("run-a", "inv-a", True, "confirmed"),
                                 ("run-r", "inv-r", False, "")]
    assert episode_runs(rows, None) == rows and episode_runs(None, {"runs": rows}) == rows, "a list merges as itself"


def _settled_run(root, task_id, run_id, *, terminal_state="cancelled", source_bytes=None, corrupt=False, **facts):
    from ouroboros import delegate_custody_current as current, store
    from ouroboros.artifacts import store_actor_source_bytes

    detail = {"lastSeq": 3, "summary": {"state": terminal_state, "runId": run_id}}
    data = source_bytes if source_bytes is not None else json.dumps(detail).encode("utf-8")
    ref = {**store_actor_source_bytes(root, task_id, category="tool_results", source_id=f"delegated-terminal-{run_id}",
                                      data=data, extension="json"), "task_id": task_id}
    if corrupt:
        ref = {**ref, "sha256": hashlib.sha256(b"other bytes").hexdigest()}
    with store.open_store(root, write=True) as txn:
        txn.put_custody(run_id, {"run_id": run_id, "task_id": task_id, "route_id": "codex", "settled": True,
                                 "terminal_state": terminal_state, "terminal_source": ref, **facts},
                        state=current.CLOSED)


def test_every_unknown_is_named_and_a_natural_completion_is_not_called_a_stop(tmp_path, monkeypatch):
    from ouroboros import delegate_custody as dc
    from ouroboros import delegate_custody_current as current, store
    from ouroboros.external_runs import episode_disclosure

    _engine_offline(monkeypatch)
    _settled_run(tmp_path, "t", "run-done", terminal_state="succeeded", output_artifact="out.json",
                 output_complete=True, output_consumed=True, patch_captured=True, invocation_id="inv-done")
    _settled_run(tmp_path, "t", "run-bad", corrupt=True)
    with store.open_store(tmp_path, write=True) as txn:  # the engine no longer had it: closed absent
        txn.put_custody("run-gone", {"run_id": "run-gone", "task_id": "t", "settled": True}, state=current.CLOSED)
        txn.put_custody("run-review", {"run_id": "run-review", "task_id": "t", "source": "review_substrate",
                                       "route_id": "codex"}, state=current.HELD)
    assert dc.record_start_requested(tmp_path, run_id="", task_id="t", invocation_id="inv-wait",
                                     idempotency_key="inv-wait", max_seconds=60, request={"prompt": "x"},
                                     project_id="p", project_owned=False, route="claude")
    participants = [
        {"invocation_id": "inv-done", "stop_requested": True, "stop_outcome": "requested"},  # bound since the park
        {"run_id": "run-bad", "stop_requested": True, "stop_outcome": "confirmed"},
        {"run_id": "run-gone", "stop_requested": True, "stop_outcome": "requested"},
        {"run_id": "run-review", "review_owned": True},
        {"invocation_id": "inv-wait"},
        {"run_id": "run-missing", "stop_requested": True, "stop_outcome": "error:Timeout"},
    ]
    lines = dict(line.split(":", 1) for line in episode_disclosure(tmp_path, "t", participants).splitlines())
    done = lines["- run run-done (codex)"]
    assert "engine terminal: succeeded" in done and "it finished on its own; the stop did not end it" in done
    assert "the engine reported no resumable block" in done
    assert "output retained (complete, read)" in done and "patch captured, awaiting apply or reject" in done
    bad = lines["- run run-bad (codex)"]
    assert "failed local verification (ValueError)" in bad and 'read_file(root="artifact_store", path="' in bad
    assert "it may be continued" in bad, "a terminal observed is continuable even when its retained copy is damaged"
    assert "custody closed it without a recorded engine terminal" in lines["- run run-gone"]
    assert "continue_from" not in lines["- run run-gone"]
    review = lines["- run run-review (codex)"]
    assert "its panel owns it" in review and "continue_from" not in review and "delegate_wait" not in review
    assert 'delegate_start(retry_of="inv-wait") retrieves it without starting another' in lines["- start inv-wait"]
    assert "custody holds no row for it now: its fate is UNKNOWN" in lines["- run run-missing"]


def test_an_unreadable_store_a_record_without_a_list_and_a_long_list_are_disclosed(tmp_path, monkeypatch):
    from ouroboros import delegate_custody_current
    from ouroboros.external_runs import _DETAILED, episode_disclosure, paused_runs_notice

    ctx = SimpleNamespace(task_id="t", drive_root=tmp_path, budget_drive_root=tmp_path)
    many = [{"run_id": f"run-{index}"} for index in range(_DETAILED + 3)]
    text = episode_disclosure(tmp_path, "t", many)
    assert text.splitlines()[-1].startswith(f"- 3 more, not detailed here: run run-{_DETAILED}, ")
    assert len(text.splitlines()) == _DETAILED + 1
    legacy = paused_runs_notice(ctx, {"reason": "owner"}, {"runs": []})
    assert legacy.splitlines() == [PARTIAL, "- none named"]

    def broken(*_a, **_k):
        raise OSError("disk gone")

    monkeypatch.setattr(delegate_custody_current, "task_snapshot", broken)
    assert episode_disclosure(tmp_path, "t", many[:1]) == (
        '- custody unreadable now (OSError): what became of run run-0 is UNKNOWN; '
        'delegate_wait(run_id="...") reads each run')


def test_a_budget_pause_resume_keeps_its_own_census_disclosure(tmp_path, monkeypatch):
    """Owner Pause only: an ordinary budget pause (no owner Pause ever recorded participants on it:
    the settle tick never writes a budget row) keeps its census Resume text unchanged."""
    from ouroboros import budget_pause
    from ouroboros.external_runs import owner_pause_runs

    ctx, _limit = _loop_ctx(tmp_path, "money")
    ctx._budget_paused_sec = 0.0
    messages: list = []
    row = {"reason": "budget", "rail": "global_exhausted",
           "external_runs": {"runs": [{"run_id": "run-1", "state": "stop_confirmed", "stop_outcome": "confirmed"}]}}
    assert owner_pause_runs(ctx, row) == ""
    budget_pause._resume_disclosure(ctx, row, {}, {"consumed_at": 1.0}, {"reason": "hard_rail"}, messages)
    text = messages[-1]["content"]
    assert "Delegated runs this task holds (re-observed at this Resume):\n- run run-1: stop_confirmed" in text
    assert "its participants" not in text


# --- R1 repairs: a racing same-wait write and a record without a full list -------------------

PARTIAL = "- this Pause's participant list is partial: a run it stopped that settled before the list began is not named here"


def test_a_tick_write_landing_while_the_direct_actor_resumes_is_kept(tmp_path, monkeypatch):
    """R1: the settle tick binds and stops a late start on the SAME warm wait between the direct
    actor's decision to resume and its ``resumed`` write. That write transitions the CURRENT wait
    in one update, so the first resumed notice names the late stop instead of dropping it."""
    from ouroboros import delegate_custody as dc, owner_wait
    from ouroboros.task_results import load_task_result
    from supervisor import owner_pause_control

    remote = {"state": "running"}
    queue, ctx, limit = _warm_author(tmp_path, monkeypatch, remote)
    real_set, raced = owner_wait.set_owner_wait, {}

    def resumed_write_after_a_tick(root, task_id, wait, *args, **kwargs):
        resumed = callable(wait) or (isinstance(wait, dict) and wait.get("state") == "resumed")
        if resumed and not raced:
            raced["at"] = load_task_result(tmp_path, "author")["owner_wait"]["state"]
            assert dc.record_started(tmp_path, dc.RunCustody(
                run_id="run-late", task_id="author", route_id="claude", model="m", project_id="p",
                project_owned=False, root_task_id="author", ledger_root=str(tmp_path), invocation_id="inv-late"))
            owner_pause_control._reread_warm_member(queue, "author", {
                "task": {"budget_drive_root": str(tmp_path)}, "owner_wait": raced_wait()})
        return real_set(root, task_id, wait, *args, **kwargs)

    def raced_wait():
        return load_task_result(tmp_path, "author")["owner_wait"]

    monkeypatch.setattr(owner_wait, "set_owner_wait", resumed_write_after_a_tick)
    box = _park_warm_then(tmp_path, monkeypatch, remote, queue, ctx, limit, resume=True)
    assert box["cause"] == "control:owner_resume" and raced["at"] == "waiting", (box, raced)
    durable = load_task_result(tmp_path, "author")["owner_wait"]
    assert durable["state"] == "resumed" and durable["resume_reason"] == "control:owner_resume"
    assert [(row["run_id"], row["stop_outcome"]) for row in durable["owner_pause"]["episode_runs"]] == [
        ("run-own", "confirmed"), ("run-late", "confirmed")], "the racing tick's fact survives the transition"
    notice = next(str(message["content"]) for message in reversed(limit.messages)
                  if "has now resumed it" in str(message.get("content") or ""))
    late = next(text for text in notice.splitlines() if text.startswith("- run run-late"))
    assert "this Pause requested its stop (stop outcome: confirmed)" in late


def _paused_solo(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from supervisor.owner_pause_control import request_owner_pause

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    write_task_result(tmp_path, "solo", STATUS_RUNNING, chat_id=0)
    workers.RUNNING["solo"] = {"task": {"id": "solo", "type": "task", "chat_id": 0, "root_task_id": "solo"},
                               "worker_id": 0, "attempt": 1}
    _own_run(tmp_path, "solo")
    remote = {"state": "running"}
    _daemon(monkeypatch, remote)
    assert request_owner_pause("solo", request_id="press")["ok"]
    return queue, workers, remote


def test_a_cold_pause_written_before_its_list_keeps_its_gap_while_new_facts_join(tmp_path, monkeypatch):
    """R1: an owner pause row an older body wrote has no list. The settle tick adds what it now
    sees, never claiming the list is whole: the Resume names those facts AND keeps the gap."""
    from ouroboros import budget_pause
    from ouroboros.external_runs import paused_runs_notice
    from supervisor import owner_pause_control

    queue, workers, remote = _paused_solo(tmp_path, monkeypatch)
    parked = _cold_park(tmp_path, monkeypatch, queue, workers)
    legacy = {key: value for key, value in parked.items() if key != "episode_runs"}
    budget_pause.set_budget_pause(tmp_path, "solo", legacy, expected_pause_id=parked["pause_id"])
    remote["state"] = "cancelled"
    monkeypatch.setattr(owner_pause_control, "_LAST_SETTLE_CHECK", {})
    assert owner_pause_control.settle_requested_owner_pauses(queue, now=100.0) == ["solo"]
    row = budget_pause.budget_pause_row(tmp_path, "solo")
    assert [(item["run_id"], item["stop_outcome"]) for item in row["episode_runs"]] == [("run-own", "confirmed")]
    assert row.get("episode_runs_partial") is True
    ctx, _limit = _loop_ctx(tmp_path, "solo")
    _engine_offline(monkeypatch)
    lines = paused_runs_notice(ctx, row).splitlines()
    assert lines[0] == PARTIAL and lines[1].startswith("- run run-own (codex): this Pause requested its stop")


def test_a_warm_park_without_a_list_stays_partial_through_its_tick_and_cold_conversion(tmp_path, monkeypatch):
    """R1: a warm park an older body wrote has no list; its tick and its Restart conversion keep
    the gap beside the new facts, and the cold Resume reader shows both."""
    from ouroboros.external_runs import paused_runs_notice
    from ouroboros.owner_wait import set_owner_wait
    from supervisor.restart_retention import retain_owner_pause_park

    remote = {"state": "running"}
    queue, ctx, limit = _warm_author(tmp_path, monkeypatch, remote)

    def as_an_older_body_wrote_it(parked):
        pause = {key: value for key, value in parked["owner_pause"].items() if key != "episode_runs"}
        set_owner_wait(tmp_path, "author", {**parked, "owner_pause": pause},
                       expected_wait_id=parked["wait_id"], expected_state="waiting")

    box = _park_warm_then(tmp_path, monkeypatch, remote, queue, ctx, limit, resume=False,
                          before_tick=as_an_older_body_wrote_it)
    recorded = box["recorded"]["owner_pause"]
    assert [(row["run_id"], row["stop_outcome"]) for row in recorded["episode_runs"]] == [("run-own", "confirmed")]
    assert recorded.get("episode_runs_partial") is True
    cold = retain_owner_pause_park(tmp_path, "author", 1)
    assert cold["episode_runs"] == recorded["episode_runs"] and cold.get("episode_runs_partial") is True
    _engine_offline(monkeypatch)
    lines = paused_runs_notice(ctx, cold).splitlines()
    assert lines[0] == PARTIAL and lines[1].startswith("- run run-own (codex): this Pause requested its stop")


def test_a_pause_completed_from_saved_work_after_a_crash_is_partial(tmp_path, monkeypatch):
    """R1: a seed whose worker died may already have stopped runs that settled; the completion's own
    observation cannot see them, so its list is partial rather than whole."""
    from ouroboros import budget_pause, working_checkpoint as wc
    from ouroboros.external_runs import paused_runs_notice
    from ouroboros.task_results import write_task_result

    _ctx, limit_ctx = _loop_ctx(tmp_path, "crashed")
    write_task_result(tmp_path, "crashed", "running")
    wc.save_round(limit_ctx, "pre_effect")
    seed = {"pause_id": "p-owner", "state": budget_pause.STATE_PAUSING, "reason": "owner", "rail": "owner_pause",
            "pause_generation": 1, "scope": "root", "task_attempt": 1, "source_ref": None}
    budget_pause.set_budget_pause(tmp_path, "crashed", seed)
    row = wc.complete_pause_from_working(tmp_path, "crashed", seed, 1)
    assert row["episode_runs"] == [] and row.get("episode_runs_partial") is True
    assert paused_runs_notice(_ctx, row).splitlines() == [PARTIAL, "- none named"]


def test_the_warm_notice_reader_keeps_a_park_without_a_list_partial():
    """R1: ``_parked_owner_pause`` merges later facts into a park that kept no list without making it whole."""
    from types import SimpleNamespace as NS

    from ouroboros import owner_wait

    ctx = NS(task_id="nobody", drive_root="/nonexistent-root", budget_drive_root="/nonexistent-root")
    merged = owner_wait._parked_owner_pause(ctx, {"wait_id": "w"}, {"external_runs": {"runs": []}})
    assert merged.get("episode_runs_partial") is True
    complete = owner_wait._parked_owner_pause(ctx, {"wait_id": "w"}, {"episode_runs": []})
    assert "episode_runs_partial" not in complete
