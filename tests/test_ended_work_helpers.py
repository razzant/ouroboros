"""#3047: a helper still paused when its work truly ended closes with that work; nothing else is touched.

The real exact pause, the queue, the cancel-intent watchdog, custody, ``task_done`` dispatch and the server's work
projection run here; only the time between the discovery and the watchdog's next pass is skipped.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from ouroboros import cancel_intents as ci
from ouroboros.budget_pause import budget_pause_row
from ouroboros.owner_continue import binding_sha
from ouroboros.post_task_checkpoint import POST_TASK_SYNTHESIS_OPEN_STATUSES
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.work_stop import SELECTION_KEY, stop_record
from supervisor.ended_work_helpers import WORK_FINISHED, close_abandoned_helpers
from supervisor.task_lifecycle import sweep_cancel_intents
from tests._budget_pause_exact_helpers import _install_queue, _parked, _supervisor_ctx

ROOT, CHILD = "work-root", "work-child"


def _work_root(root, task_id=ROOT, status="completed", **fields):
    write_task_result(root, task_id, "running", chat_id=7, root_task_id=task_id, delegation_role="root", title="A work")
    write_task_result(root, task_id, status, result="the answer", **fields)


def _paused_helper(root, monkeypatch, task_id=CHILD, *, extra=None, scope="global", work=ROOT):
    lineage = {"parent_task_id": work, "delegation_role": "subagent", "role": "researcher", "chat_id": 7}
    task, _row = _parked(root, monkeypatch, task_id=task_id, scope=scope, root_task_id=work, extra={**lineage, **(extra or {})})
    write_task_result(root, task_id, "scheduled", root_task_id=work, **lineage)
    return task


@pytest.fixture
def env(tmp_path, monkeypatch):
    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    events = []
    monkeypatch.setattr(workers, "get_event_q", lambda *a, **k: SimpleNamespace(put=events.append))
    return SimpleNamespace(q=queue, workers=workers, root=tmp_path, events=events, monkeypatch=monkeypatch)


def _sweep(env):
    """One discovery pass, then the watchdog's next pass over what it minted."""
    minted = close_abandoned_helpers()
    outcomes = sweep_cancel_intents(now=time.time() + 3600)
    return minted, outcomes


def _pending(env, task_id=CHILD):
    return next((row for row in env.workers.PENDING if row["id"] == task_id), None)


def _closed(env, task_id=CHILD):
    row = load_task_result(env.root, task_id) or {}
    assert row.get("status") == "cancelled", row
    assert row["cancel_origin"]["source"] == WORK_FINISHED
    assert "parent_decision" not in row  # the parent decided nothing
    assert _pending(env, task_id) is None
    assert not ci.active_intent(env.root, task_id)
    return row


def _retained(env, task_id=CHILD):
    assert _pending(env, task_id) is not None
    assert not ci.active_intent(env.root, task_id)
    assert (load_task_result(env.root, task_id) or {}).get("status") != "cancelled"


AUTHOR_STOP = {"task_completion": {"action": "stop", "rationale": "handed on", "source": "finish_task"}}


@pytest.mark.parametrize("status,fields", [
    ("completed", {}), ("completed", AUTHOR_STOP), ("failed", {"reason_code": "provider_failure"}),
    ("cancelled", {"cancel_origin": {"source": "agent_tool", "reason": "enough"}}),
], ids=["completed", "author_stop", "failed", "stopped"])
def test_every_ordinary_end_of_the_root_closes_its_exact_paused_helper(env, status, fields):
    _work_root(env.root, status=status, **fields)
    _paused_helper(env.root, env.monkeypatch)
    minted, outcomes = _sweep(env)
    assert minted == [CHILD] and outcomes == {CHILD: "cancelled"}
    row = _closed(env)
    # The stored pause that was never resumed stays readable beside the closed result.
    assert row["pause_at_close"]["_budget_pause"]["exact_continuation"] is True
    assert budget_pause_row(env.root, CHILD)["source_ref"]  # the exact saved source remains addressable
    assert load_task_result(env.root, ROOT)["status"] == status and load_task_result(env.root, ROOT)["result"] == "the answer"


@pytest.mark.parametrize("hold", [
    "saved_work_hold", "owner_restart_hold", "saved_sleep_recovery", "root_fence_member_pending_selection", "authority",
])
def test_every_hold_that_offers_resume_closes_with_the_ended_work(env, hold):
    _work_root(env.root)
    write_task_result(env.root, CHILD, "scheduled", root_task_id=ROOT, parent_task_id=ROOT, delegation_role="subagent")
    env.workers.PENDING.append({
        "id": CHILD, "type": "task", "chat_id": 7, "root_task_id": ROOT, "parent_task_id": ROOT, "_attempt": 1,
        "delegation_role": "subagent", "_budget_pause_hold": stop_record(hold, held_at="2026-11-10T00:00:00+00:00")})
    assert _sweep(env)[0] == [CHILD]
    assert _closed(env)["pause_at_close"]["_budget_pause_hold"]["reason"] == hold


def test_a_root_without_task_done_is_read_from_its_durable_end(env):
    """A Presence turn settles its own row; no queue row or ``task_done`` of it ever reaches this pass."""
    _work_root(env.root, _presence_turn={"turn_id": "turn-1"})
    _paused_helper(env.root, env.monkeypatch)
    assert _sweep(env)[0] == [CHILD]
    _closed(env)


@pytest.mark.parametrize("phase", sorted(POST_TASK_SYNTHESIS_OPEN_STATUSES))
def test_open_late_work_keeps_the_helper_until_it_settles(env, phase):
    _work_root(env.root, root_phase_checkpoint={"post_task_synthesis": phase})
    _paused_helper(env.root, env.monkeypatch)
    assert _sweep(env) == ([], {})
    _retained(env)
    write_task_result(env.root, ROOT, "completed", root_phase_checkpoint={"post_task_synthesis": "completed"})
    assert _sweep(env)[0] == [CHILD]
    _closed(env)


def test_a_live_root_or_its_running_retry_keeps_the_helper(env):
    from tests._cancel_intents_shared import _write_root_retry_pair

    _write_root_retry_pair(env.root, ROOT, "work-retry", new_status="running")
    _paused_helper(env.root, env.monkeypatch)
    env.workers.RUNNING["work-retry"] = {"task": {"id": "work-retry", "root_task_id": ROOT}, "worker_id": 0}
    assert _sweep(env) == ([], {})
    _retained(env)
    env.workers.RUNNING.clear()
    write_task_result(env.root, "work-retry", "completed", result="retry answer")
    assert _sweep(env)[0] == [CHILD]
    _closed(env)


def test_a_running_member_of_the_tree_keeps_its_paused_sibling(env):
    _work_root(env.root)
    _paused_helper(env.root, env.monkeypatch)
    env.workers.RUNNING["other-child"] = {"task": {"id": "other-child", "root_task_id": ROOT}, "worker_id": 0}
    assert _sweep(env) == ([], {})
    _retained(env)


@pytest.mark.parametrize("sleeper", [ROOT, "sleeping-child"], ids=["root", "sibling"])
def test_a_cold_sleep_keeps_the_tree_live_until_the_sleeper_ends(env, sleeper):
    """Real sleep arming, exact parking and queue install: releasing a process does not end its work."""
    from ouroboros import budget_pause as bp, model_sleep
    from supervisor.events_budget import install_exact_budget_pause
    from tests._budget_pause_exact_helpers import _fast_hold, _loop_ctx, _quiet_external

    _work_root(env.root, status="running" if sleeper == ROOT else "completed")
    _paused_helper(env.root, env.monkeypatch)
    lineage = {} if sleeper == ROOT else {"parent_task_id": ROOT, "delegation_role": "subagent"}
    write_task_result(env.root, sleeper, "running", root_task_id=ROOT, chat_id=7, **lineage)
    task = {"id": sleeper, "type": "task", "chat_id": 7, "root_task_id": ROOT, "_attempt": 1, **lineage}
    env.workers.RUNNING[sleeper] = {"task": task, "worker_id": None, "attempt": 1}
    ctx, limit = _loop_ctx(env.root, sleeper)
    ctx.root_task_id = ROOT
    ctx.task_metadata = {"root_task_id": ROOT, **lineage}
    _fast_hold(env.monkeypatch, bp)
    _quiet_external(env.monkeypatch, bp)
    assert model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, wake_after_sec=3600), "cold")["reason"] == "sleep_armed"
    try:
        with pytest.raises(bp.BudgetPauseRequested) as raised:
            bp.enter_cold_sleep(limit)
    finally:
        bp.end_dispatch_fence(sleeper)
    install_exact_budget_pause(_supervisor_ctx(env.root, env.workers, env.q, [], []), sleeper,
                               bp.exact_pause_marker(raised.value.pause)["checkpoint"])
    assert sleeper not in env.workers.RUNNING and _pending(env, sleeper)["_budget_pause"]["reason"] == "sleep"
    assert _sweep(env) == ([], {})
    _retained(env)
    _retained(env, sleeper)
    env.workers.PENDING.remove(_pending(env, sleeper))
    write_task_result(env.root, sleeper, "completed", result="waiting work finished")
    assert _sweep(env)[0] == [CHILD]
    _closed(env)


def _technical(env, task_id=ROOT):
    _work_root(env.root, task_id=task_id, status="cancelled",
               cancel_origin={"source": "snapshot_restore", "reason": "server_shutdown"})


def _continued(env, predecessor, successor, *, state="admitted"):
    binding = {"predecessor_task_id": predecessor, "successor_task_id": successor, "action_nonce": "press-0001-abcdef"}
    write_task_result(env.root, predecessor, "cancelled", continued_by={
        "by": {}, "successor_task_id": successor, "action_nonce": "press-0001-abcdef", "binding": binding,
        "binding_sha256": binding_sha(binding), "state": state})


def _successor(env, successor, predecessor, status):
    write_task_result(env.root, successor, status, root_task_id=successor, delegation_role="root", result="next",
                      metadata={"continuation": {"predecessor_task_id": predecessor, "admission": {"by": {}}}})


def test_a_technical_stop_alone_keeps_the_helper(env):
    _technical(env)
    _paused_helper(env.root, env.monkeypatch)
    assert _sweep(env) == ([], {})
    _retained(env)


def test_a_bound_only_continue_is_not_an_ending(env):
    _technical(env)
    _paused_helper(env.root, env.monkeypatch)
    _continued(env, ROOT, "work-next", state="bound")
    assert _sweep(env) == ([], {})
    _retained(env)


@pytest.mark.parametrize("successor_status", ["running", "completed", None], ids=["live", "ended", "unwritten"])
def test_a_technical_stops_own_admitted_continue_ends_it_whatever_the_successor_does(env, successor_status):
    """Continue cannot steer a predecessor's helpers, so nothing waits for the new root."""
    _technical(env)
    _paused_helper(env.root, env.monkeypatch)
    _continued(env, ROOT, "work-next")
    if successor_status:
        _successor(env, "work-next", ROOT, successor_status)
    assert _sweep(env)[0] == [CHILD]
    _closed(env)


@pytest.mark.parametrize("reason", ["stale_resume_grant_superseded", "some_future_refusal"])
def test_a_refusal_that_offers_resume_closes_with_the_ended_work(env, reason):
    _work_root(env.root)
    task = _paused_helper(env.root, env.monkeypatch)
    task["_budget_pause"]["refusal"] = {"reason": reason}
    assert _sweep(env)[0] == [CHILD]
    _closed(env)


@pytest.mark.parametrize("marks", [
    {"_continuation_prepared": True}, {"_owner_hold": {"reason": "owner"}},
    {"_budget_pause_hold": {"reason": "dispatch_outcome_unknown"}},
], ids=["publication_unconfirmed", "owner_hold", "dispatch_unknown"])
def test_an_unknown_dispatch_or_an_owner_hold_keeps_the_helper(env, marks):
    _work_root(env.root)
    write_task_result(env.root, CHILD, "scheduled", root_task_id=ROOT, parent_task_id=ROOT, delegation_role="subagent")
    env.workers.PENDING.append({"id": CHILD, "type": "task", "chat_id": 7, "root_task_id": ROOT, "parent_task_id": ROOT,
                                "_attempt": 1, "delegation_role": "subagent", **marks})
    assert _sweep(env) == ([], {})
    _retained(env)


def test_a_continue_of_an_ordinary_end_protects_nothing_of_the_predecessor(env):
    _work_root(env.root, **AUTHOR_STOP)
    _paused_helper(env.root, env.monkeypatch)
    write_task_result(env.root, "work-next", "running", root_task_id="work-next", delegation_role="root")
    env.workers.RUNNING["work-next"] = {"task": {"id": "work-next", "root_task_id": "work-next"}, "worker_id": 0}
    assert _sweep(env)[0] == [CHILD]
    _closed(env)
    assert "work-next" in env.workers.RUNNING


def test_a_missing_or_unreadable_root_keeps_the_helper(env, monkeypatch):
    _paused_helper(env.root, env.monkeypatch)
    assert _sweep(env) == ([], {})
    _retained(env)
    _work_root(env.root)
    import supervisor.task_ownership as ownership
    monkeypatch.setattr(ownership.TaskOwnershipRead, "load", lambda self, task_id: (_ for _ in ()).throw(OSError("unreadable")))
    assert _sweep(env) == ([], {})
    _retained(env)


def test_an_owner_pause_panic_hold_or_accepted_resume_keeps_the_helper(env):
    _work_root(env.root)
    task = _paused_helper(env.root, env.monkeypatch)
    pause = {key: task.pop(key) for key in ("_budget_pause",)}
    write_task_result(env.root, ROOT, "completed", owner_pause={"state": "paused", "fence_id": "fence-1"})
    task.update(pause)
    assert _sweep(env) == ([], {})
    _retained(env)
    write_task_result(env.root, ROOT, "completed", owner_pause={"state": "released", "fence_id": "fence-1"})
    held = dict(task)
    task["_budget_pause_hold"] = stop_record("panic_hold", held_at="2026-11-10T00:00:00+00:00")
    assert _sweep(env) == ([], {})
    _retained(env)
    task.pop("_budget_pause_hold")
    marker = task.pop("_budget_pause")  # an accepted Resume's carrier lifts the stop it answered
    for carrier in ({"_budget_pause_resume": {"grant_id": "grant-1"}}, {SELECTION_KEY: {"selected_by": "owner"}}):
        task.update(carrier)
        assert _sweep(env) == ([], {})
        _retained(env)
        task.pop(next(iter(carrier)))
    task["_budget_pause"] = marker  # each guard above is the one that held it: with none left, it closes
    assert task.keys() >= held.keys()
    assert _sweep(env)[0] == [CHILD]
    _closed(env)


def test_panic_closes_nothing(env, monkeypatch):
    import supervisor.restart_retention as retention

    _work_root(env.root)
    _paused_helper(env.root, env.monkeypatch)
    monkeypatch.setattr(retention, "panic_recorded", lambda _root: True)
    assert close_abandoned_helpers() == []
    _retained(env)


def test_intent_first_refuses_the_resume_and_the_closure_stands(env):
    from supervisor.queue_transitions import resume_budget_paused_task

    _work_root(env.root)
    _paused_helper(env.root, env.monkeypatch)
    assert close_abandoned_helpers() == [CHILD]
    assert resume_budget_paused_task(CHILD)["ok"] is False
    sweep_cancel_intents(now=time.time() + 3600)
    _closed(env)


def test_resume_first_withdraws_the_closure_and_the_row_stays_resumed(env):
    _work_root(env.root)
    task = _paused_helper(env.root, env.monkeypatch)
    assert close_abandoned_helpers() == [CHILD]
    task.pop("_budget_pause")  # the Resume won between discovery and custody: its carrier lifts that stop
    task["_budget_pause_resume"] = {"grant_id": "grant-1"}
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: "not_found"}
    _retained(env)
    assert isinstance(_pending(env)["_budget_pause_resume"], dict)


def test_a_carrier_a_newer_pause_spent_holds_nothing(env):
    """A Resume's carrier answers the stop it lifted. A pause marker standing over it is a newer Pause (a re-park copies
    the old carrier along), so the parked helper of the ended work still closes: that carrier is no accepted Resume."""
    _work_root(env.root)
    task = _paused_helper(env.root, env.monkeypatch)
    task.update({SELECTION_KEY: {"selected_by": "owner"}, "_budget_pause_resume": {"grant_id": "grant-0"}})
    assert _sweep(env)[0] == [CHILD]
    _closed(env)


def test_capture_runs_before_the_cancelled_write_and_natural_completion_wins(env, monkeypatch):
    import supervisor.cancel_publication as publication

    _work_root(env.root)
    _paused_helper(env.root, env.monkeypatch)
    seen = {}

    def salvage(q, task, task_id):
        seen["status"] = load_task_result(env.root, task_id)["status"]
        write_task_result(env.root, task_id, "completed", result="finished by itself")  # the writer's guard keeps this
        return "", "", ""

    monkeypatch.setattr(publication, "_salvage_cancelled_output", salvage)
    assert _sweep(env)[1] == {CHILD: "already_settled"}
    assert seen["status"] == "scheduled"
    assert load_task_result(env.root, CHILD)["result"] == "finished by itself"
    assert _pending(env) is None and not ci.active_intent(env.root, CHILD)


def test_workspace_patch_and_external_custody_are_taken_before_the_close(env, monkeypatch):
    import ouroboros.headless as headless
    import supervisor.cancel_publication as publication

    _work_root(env.root)
    _paused_helper(env.root, env.monkeypatch, extra={"metadata": {"workspace_root": str(env.root)}, "admitted_dispatch": "possible"})
    order = []
    real_write = write_task_result

    def finalize(root, task, **_):
        order.append(("capture", load_task_result(root, CHILD)["status"]))
        real_write(root, CHILD, load_task_result(root, CHILD)["status"], artifact_status="ready_no_changes")
        return []

    def audit(q, task_id, **kwargs):
        order.append(("audit", load_task_result(env.root, task_id)["status"]))
        return {"unreconciled": ["run-open"], "audit_status": "ok", "outcomes": []}

    monkeypatch.setattr(headless, "finalize_task_artifacts", finalize)
    monkeypatch.setattr(publication, "_audit_delegated_runs_on_kill", audit)
    assert _sweep(env)[0] == [CHILD]
    row = _closed(env)
    assert order == [("capture", "scheduled"), ("audit", "cancelled")]  # the irreversible audit after the committed write
    assert row["artifact_status"] == "ready_no_changes"
    # An external run that could not be proven stopped stays disclosed on the closed result.
    assert row["delegated_runs_unreconciled"] == ["run-open"]


def test_a_real_workspace_patch_is_captured_before_the_cancelled_write(env, tmp_path_factory):
    """No mock of the capture: a real repository whose saved edit the helper never got to hand in."""
    import subprocess

    workspace = tmp_path_factory.mktemp("helper-workspace")
    def git(*args):
        subprocess.run(["git", "-C", str(workspace), "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True, capture_output=True)

    git("init", "-q")
    (workspace / "note.txt").write_text("base\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-qm", "base")
    base = subprocess.run(["git", "-C", str(workspace), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    (workspace / "note.txt").write_text("base\nsaved by the paused helper\n", encoding="utf-8")

    _work_root(env.root)
    meta = {"workspace_root": str(workspace), "workspace_mode": "self_worktree", "workspace_base_sha": base}
    _paused_helper(env.root, env.monkeypatch, extra={"metadata": meta, "workspace_root": str(workspace), "workspace_mode": "self_worktree",
                                                     "workspace_base_sha": base, "admitted_dispatch": "possible"})
    write_task_result(env.root, CHILD, "scheduled", workspace_root=str(workspace), workspace_mode="self_worktree", workspace_base_sha=base, metadata=meta)
    assert _sweep(env)[1] == {CHILD: "cancelled"}
    row = _closed(env)
    assert row["artifact_status"] == "ready_with_changes", row
    from ouroboros.headless import task_artifacts_dir

    patches = list(task_artifacts_dir(env.root, CHILD).rglob("*.patch"))
    manifests = list(task_artifacts_dir(env.root, CHILD).rglob("workspace_patch.json"))
    assert manifests, list(task_artifacts_dir(env.root, CHILD).rglob("*"))
    assert any("saved by the paused helper" in patch.read_text(encoding="utf-8") for patch in patches), (patches, manifests[0].read_text(encoding="utf-8"))


def test_a_failed_capture_returns_the_row_and_the_watchdog_finishes_it(env, monkeypatch):
    import ouroboros.headless as headless

    _work_root(env.root)
    _paused_helper(env.root, env.monkeypatch, extra={"metadata": {"workspace_root": str(env.root)}, "admitted_dispatch": "possible"})
    calls = {"n": 0}

    def flaky(root, task, **_):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full")
        write_task_result(root, CHILD, load_task_result(root, CHILD)["status"], artifact_status="ready_no_changes")
        return []

    monkeypatch.setattr(headless, "finalize_task_artifacts", flaky)
    assert _sweep(env)[1] == {CHILD: "failed"}
    assert _pending(env) is not None and ci.active_intent(env.root, CHILD)
    assert load_task_result(env.root, CHILD)["status"] == "scheduled"
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: "cancelled"}
    _closed(env)


def test_a_closed_helper_lets_the_last_root_fence_go_and_its_card_settles_after_task_done(env):
    from supervisor import events

    _work_root(env.root)
    _paused_helper(env.root, env.monkeypatch, scope="root")
    assert ROOT in env.q.BUDGET_ROOT_FENCES
    assert _sweep(env)[0] == [CHILD]
    _closed(env)
    [event] = [row for row in env.events if row.get("type") == "task_done"]
    sent, pushed = [], []
    ctx = _supervisor_ctx(env.root, env.workers, env.q, [], pushed)
    ctx.send_with_budget = lambda chat, text, **kw: sent.append((chat, text, kw))
    ctx.append_jsonl = lambda *a, **k: True
    ctx.owner_chat_id = 7
    events._handle_task_done(event, ctx)
    assert ROOT not in env.q.BUDGET_ROOT_FENCES
    [(chat, text, frame)] = sent
    meta = frame["progress_meta"]
    assert (meta["subagent_event"], meta["subagent_task_id"], meta["parent_task_id"], meta["root_task_id"]) == (
        "cancelled", CHILD, ROOT, ROOT)
    assert chat == 7 and "cancelled" in text
    assert pushed[-1]["cancel_origin"]["source"] == WORK_FINISHED


def test_a_replayed_closure_notice_stays_one_ending(env):
    """The real ``task_done`` delivered twice, its queue row gone both times: one durable ending, one identical notice."""
    from supervisor import events

    _work_root(env.root)
    _paused_helper(env.root, env.monkeypatch)
    assert _sweep(env)[0] == [CHILD]
    [event] = [row for row in env.events if row.get("type") == "task_done"]
    sent, pushed = [], []
    ctx = _supervisor_ctx(env.root, env.workers, env.q, [], pushed)
    ctx.send_with_budget = lambda chat, text, **kw: sent.append((chat, text, kw))
    ctx.append_jsonl = lambda *a, **k: True
    ctx.owner_chat_id = 7
    before = load_task_result(env.root, CHILD)
    events._handle_task_done(dict(event), ctx)
    events._handle_task_done(dict(event), ctx)
    assert sent and len({(chat, text, kw["progress_meta"]["subagent_task_id"]) for chat, text, kw in sent}) == 1, sent
    after = load_task_result(env.root, CHILD)
    assert (after["status"], after["cancel_origin"], after["result"]) == (before["status"], before["cancel_origin"], before["result"])


def test_the_closed_helper_leaves_no_amber_and_the_next_root_stays_visible(env):
    from tests._work_projection_shared import state_fields

    _work_root(env.root, **AUTHOR_STOP)
    _paused_helper(env.root, env.monkeypatch)
    write_task_result(env.root, "work-next", "running", root_task_id="work-next", delegation_role="root", title="Next",
                      chat_id=7, metadata={"continuation": {"predecessor_task_id": ROOT, "work_id": ROOT, "work_name": "A work",
                                                            "admission": {"by": {"actor": "owner"}}}})
    env.workers.RUNNING["work-next"] = {"task": {"id": "work-next", "type": "task", "chat_id": 7, "root_task_id": "work-next"},
                                        "worker_id": 0, "attempt": 1, "started_at": time.time(), "last_heartbeat_at": time.time()}
    env.q.persist_queue_snapshot(reason="fixture")
    before = state_fields(env.root)
    assert _turn(before), "the control: the paused helper holds the owner's turn before the closure"
    _sweep(env)
    _closed(env)
    env.q.persist_queue_snapshot(reason="fixture")
    after = state_fields(env.root)
    assert not _turn(after)
    assert not any(room.get("amber") for room in (after.get("rooms") or {}).values())
    assert any(work["activity"] == "working" for work in after["works"])


def _turn(fields):
    return [work for work in fields["works"] if work.get("turn")]


def test_the_existing_off_loop_cancel_duty_runs_the_discovery(env, monkeypatch):
    from ouroboros import server_maintenance

    _work_root(env.root)
    _paused_helper(env.root, env.monkeypatch)
    assert server_maintenance._run_terminal_duty(env.root, "cancel_intent_sweep", SimpleNamespace(release=lambda: None)) is None
    assert ci.active_intent(env.root, CHILD)["source"] == WORK_FINISHED
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: "cancelled"}
    _closed(env)


SIBLING = "work-child-2"


def _press_pause(env, request_id="press-1"):
    from supervisor.owner_pause_control import request_owner_pause

    answer = request_owner_pause(ROOT, request_id=request_id)
    assert answer["ok"] is True, answer
    return answer


def _fire_before_write(env, monkeypatch, moment, action):
    """Run ``action`` once at ``moment``: the last step before the write, or inside its preparation (no recheck is left)."""
    import ouroboros.task_custody as custody
    import supervisor.task_lifecycle as lifecycle

    module, name = {"before_write": (lifecycle, "_cancel_result_fields"), "inside_write": (custody, "capture_unread_mail")}[moment]
    real, fired = getattr(module, name), []

    def fire(*args, **kwargs):
        if not fired:
            fired.append(True)
            action()
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, fire)
    return fired


def _withdrawn(env, task_id=CHILD):
    _retained(env, task_id)
    assert (load_task_result(env.root, task_id) or {}).get("cancel_origin") in (None, {})


def test_a_real_owner_pause_after_discovery_withdraws_the_closure(env):
    _work_root(env.root)
    _paused_helper(env.root, env.monkeypatch)
    assert close_abandoned_helpers() == [CHILD]
    _press_pause(env)
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: "not_found"}
    _withdrawn(env)
    assert close_abandoned_helpers() == []  # the owner's Pause holds it, and no closure is asked again meanwhile
    _withdrawn(env)
    write_task_result(env.root, ROOT, "completed", owner_pause={"state": "released", "fence_id": "fence-1"})
    env.q.BUDGET_ROOT_FENCES.pop(ROOT, None)
    assert _sweep(env)[0] == [CHILD]  # the Pause over, the ended work's helper closes after all
    _closed(env)


@pytest.mark.parametrize("moment", ["before_write", "inside_write"])
def test_a_real_owner_pause_at_the_last_moment_keeps_every_helper(env, monkeypatch, moment):
    """The sibling still queued is what lets the real Pause admit an answered root while the first is in custody."""
    import ouroboros.process_scope as scope

    _work_root(env.root)
    _paused_helper(env.root, env.monkeypatch)
    _paused_helper(env.root, env.monkeypatch, task_id=SIBLING)
    assert set(close_abandoned_helpers()) == {CHILD, SIBLING}
    ended = []
    monkeypatch.setattr(scope, "note_task_terminal_transition", lambda _root, task_id: ended.append(task_id))
    fired = _fire_before_write(env, monkeypatch, moment, lambda: _press_pause(env))
    outcomes = sweep_cancel_intents(now=time.time() + 3600)
    assert fired and outcomes == {CHILD: "not_found", SIBLING: "not_found"}
    _withdrawn(env)
    _withdrawn(env, SIBLING)
    assert ended == []  # a refused write ends nothing: no terminal follow-up ran for it
    assert load_task_result(env.root, ROOT)["owner_pause"]["state"] in {"requested", "paused"}


@pytest.mark.parametrize("moment", ["before_write", "inside_write"])
def test_a_retry_of_the_root_published_at_the_last_moment_keeps_the_helper(env, monkeypatch, moment):
    _work_root(env.root)
    _paused_helper(env.root, env.monkeypatch)
    assert close_abandoned_helpers() == [CHILD]

    def publish_retry():
        write_task_result(env.root, ROOT, "completed", superseded_by="work-retry", retry_task_id="work-retry")
        write_task_result(env.root, "work-retry", "running", root_task_id=ROOT, parent_task_id="", delegation_role="root",
                          supersedes_task_id=ROOT, original_task_id=ROOT, timeout_retry_from=ROOT)

    fired = _fire_before_write(env, monkeypatch, moment, publish_retry)
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: "not_found"}
    assert fired
    _withdrawn(env)


@pytest.mark.parametrize("moment", ["before_write", "inside_write"])
def test_a_durable_resume_grant_at_the_last_moment_keeps_the_helper(env, monkeypatch, moment):
    from ouroboros import budget_pause

    _work_root(env.root)
    _paused_helper(env.root, env.monkeypatch)
    assert close_abandoned_helpers() == [CHILD]

    def grant():
        row = budget_pause.budget_pause_row(env.root, CHILD)
        budget_pause.set_budget_pause(env.root, CHILD, {**row, "state": budget_pause.STATE_RESUME_GRANTED})

    fired = _fire_before_write(env, monkeypatch, moment, grant)
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: "not_found"}
    assert fired
    _withdrawn(env)


def test_a_technical_stops_continue_lost_at_the_last_moment_keeps_the_helper(env, monkeypatch):
    _technical(env)
    _paused_helper(env.root, env.monkeypatch)
    _continued(env, ROOT, "work-next")
    assert close_abandoned_helpers() == [CHILD]
    fired = _fire_before_write(env, monkeypatch, "inside_write", lambda: _continued(env, ROOT, "work-next", state="bound"))
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: "not_found"}
    assert fired
    _withdrawn(env)


def test_the_closure_rechecks_with_addressed_reads_only(env, monkeypatch):
    import ouroboros.task_store as task_store

    _work_root(env.root)
    _paused_helper(env.root, env.monkeypatch)
    assert close_abandoned_helpers() == [CHILD]
    monkeypatch.setattr(task_store, "list_documents", lambda *a, **k: pytest.fail("history scanned"))
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: "cancelled"}
    _closed(env)


def test_a_resume_that_dispatched_before_the_intent_keeps_its_worker(env, monkeypatch):
    """Discovery read the row parked; an accepted Resume and its dispatch landed before the intent was written."""
    import supervisor.worker_pool_lifecycle as lifecycle
    from supervisor.queue_transitions import resume_budget_paused_task

    _work_root(env.root)
    task = _paused_helper(env.root, monkeypatch)
    real_request = ci.request_cancel
    accepted, killed = [], []

    def dispatch_first(*args, **kwargs):
        accepted.append(resume_budget_paused_task(CHILD))
        with env.q._queue_lock:
            env.q.PENDING.remove(task)
            env.q.RUNNING[CHILD] = {"task": task, "worker_id": 0, "attempt": 1}
            env.workers.WORKERS[0] = SimpleNamespace(busy_task_id=CHILD, reaping=False, proc=SimpleNamespace(
                is_alive=lambda: True, pid=123456789, join=lambda **kw: None))
        return real_request(*args, **kwargs)

    monkeypatch.setattr(ci, "request_cancel", dispatch_first)
    monkeypatch.setattr(lifecycle, "kill_worker_tree", lambda pid, **kw: killed.append(pid))
    assert close_abandoned_helpers() == [CHILD] and accepted[0]["ok"] is True, accepted
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: "not_found"}
    assert killed == [] and env.workers.WORKERS[0].reaping is False and CHILD in env.q.RUNNING
    assert load_task_result(env.root, CHILD)["status"] != "cancelled"
    assert CHILD not in ci.active_intents(env.root, strict=True)


def test_a_stop_that_joined_a_closure_is_an_ordinary_cancel_not_a_closure():
    from supervisor.ended_work_helpers import closes_only_parked_row

    assert closes_only_parked_row({"source": WORK_FINISHED})
    assert not closes_only_parked_row({"source": WORK_FINISHED, "followup_stop": {"control_id": "c1"}})
    assert not closes_only_parked_row({"source": WORK_FINISHED, "scope": "cascade"})  # a cascade widened it
    assert not closes_only_parked_row({"source": "http_single"}) and not closes_only_parked_row(None)


def test_an_owners_stop_that_joined_the_closure_still_kills_the_dispatched_worker(env, monkeypatch):
    """The closure alone yields to a dispatch (test above), but a Stop that joined it is carried out on the worker."""
    import supervisor.worker_pool_lifecycle as lifecycle
    from supervisor.queue_transitions import resume_budget_paused_task

    _work_root(env.root)
    task = _paused_helper(env.root, monkeypatch)
    real_request, alive, killed = ci.request_cancel, [True], []

    def dispatch_first(*args, **kwargs):
        assert resume_budget_paused_task(CHILD)["ok"] is True
        with env.q._queue_lock:
            env.q.PENDING.remove(task)
            env.q.RUNNING[CHILD] = {"task": task, "worker_id": 0, "attempt": 1}
            env.workers.WORKERS[0] = SimpleNamespace(wid=0, busy_task_id=CHILD, reaping=False, proc=SimpleNamespace(
                is_alive=lambda: alive[0], pid=123456789, join=lambda **kw: None))
        return real_request(*args, **kwargs)

    monkeypatch.setattr(ci, "request_cancel", dispatch_first)
    monkeypatch.setattr(lifecycle, "kill_worker_tree", lambda pid, **kw: killed.append(pid) or alive.__setitem__(0, False))
    monkeypatch.setattr(env.workers, "respawn_worker", lambda wid: None, raising=False)  # the killed slot starts no process
    assert close_abandoned_helpers() == [CHILD]
    monkeypatch.setattr(ci, "request_cancel", real_request)
    joined = real_request(env.root, CHILD, reason="the owner pressed Stop", source="http_single", requested_by="owner")
    assert joined["source"] == WORK_FINISHED and joined["already_requested"] and joined["followup_stop"]
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: "cancelled"}
    assert killed == [123456789] and load_task_result(env.root, CHILD)["status"] == "cancelled"
    assert CHILD not in ci.active_intents(env.root, strict=True)


def test_the_assignment_pass_leaves_a_closure_to_the_owner_of_its_rechecks(env):
    _work_root(env.root)
    _paused_helper(env.root, env.monkeypatch)
    assert close_abandoned_helpers() == [CHILD]
    _press_pause(env)
    with env.q._queue_lock:
        assert env.workers._drop_cancelled_pending() is True
    assert CHILD in {row["id"] for row in env.q.PENDING}
    assert load_task_result(env.root, CHILD)["status"] != "cancelled"
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: "not_found"}
    _withdrawn(env)


def test_a_started_helper_is_captured_by_the_closure_never_by_the_assignment_pass(env, monkeypatch):
    import ouroboros.headless as headless

    _work_root(env.root)
    _paused_helper(env.root, monkeypatch, extra={"metadata": {"workspace_root": str(env.root)}, "admitted_dispatch": "possible"})
    captures = []

    def finalize(root, task, **_kwargs):
        captures.append(task["id"])
        write_task_result(root, task["id"], load_task_result(root, task["id"])["status"], artifact_status="ready_no_changes")
        return []

    monkeypatch.setattr(headless, "finalize_task_artifacts", finalize)
    assert close_abandoned_helpers() == [CHILD]
    with env.q._queue_lock:
        env.workers._drop_cancelled_pending()
    assert captures == [] and load_task_result(env.root, CHILD)["status"] != "cancelled"
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: "cancelled"}
    stored = load_task_result(env.root, CHILD)
    assert stored["status"] == "cancelled" and captures == [CHILD]
    assert stored["pause_at_close"]["_budget_pause"]["exact_continuation"] is True


def _opens_of_one_pass(monkeypatch):
    from ouroboros import store

    real, opens = store.open_store, []
    monkeypatch.setattr(store, "open_store", lambda *a, **k: opens.append(1) or real(*a, **k))
    close_abandoned_helpers()
    monkeypatch.setattr(store, "open_store", real)
    return len(opens)


def test_an_unfinished_works_helpers_cost_one_verdict_and_no_row_reads(env, monkeypatch):
    import supervisor.ended_work_helpers as helpers

    _work_root(env.root, status="running")
    _paused_helper(env.root, monkeypatch)
    monkeypatch.setattr(helpers, "_parked_for_resume", lambda *a: pytest.fail("a row's control was read"))
    few = _opens_of_one_pass(monkeypatch)
    for index in range(6):
        _paused_helper(env.root, monkeypatch, task_id=f"work-helper-{index}")
    assert _opens_of_one_pass(monkeypatch) == few


def test_a_root_parked_in_the_same_pause_rests_without_a_read(env, monkeypatch):
    import supervisor.ended_work_helpers as helpers

    _work_root(env.root, status="running")
    _paused_helper(env.root, monkeypatch)
    env.q.PENDING.append({"id": ROOT, "root_task_id": ROOT, "_budget_pause": {"state": "paused"}})
    monkeypatch.setattr(helpers, "_work_ended", lambda *a: pytest.fail("the verdict was read"))
    assert close_abandoned_helpers() == []


def _close_after_a_dispatch(env, monkeypatch, task):
    """Discovery read the row parked; a Resume and its dispatch to a live worker landed before the intent was written.
    Returns the kills that worker will see."""
    import supervisor.worker_pool_lifecycle as lifecycle
    from supervisor.queue_transitions import resume_budget_paused_task

    real_request, alive, killed = ci.request_cancel, [True], []

    def dispatch_first(*args, **kwargs):
        monkeypatch.setattr(ci, "request_cancel", real_request)
        assert resume_budget_paused_task(CHILD)["ok"] is True
        with env.q._queue_lock:
            env.q.PENDING.remove(task)
            env.q.RUNNING[CHILD] = {"task": task, "worker_id": 0, "attempt": 1}
            env.workers.WORKERS[0] = SimpleNamespace(wid=0, busy_task_id=CHILD, reaping=False, proc=SimpleNamespace(
                is_alive=lambda: alive[0], pid=123456789, join=lambda **kw: None))
        return real_request(*args, **kwargs)

    monkeypatch.setattr(ci, "request_cancel", dispatch_first)
    monkeypatch.setattr(lifecycle, "kill_worker_tree", lambda pid, **kw: killed.append(pid) or alive.__setitem__(0, False))
    monkeypatch.setattr(env.workers, "respawn_worker", lambda wid: None, raising=False)
    assert close_abandoned_helpers() == [CHILD]
    return killed


def test_an_owners_stop_joining_after_the_custody_claim_is_still_carried_out(env, monkeypatch):
    """C1: the Stop lands between the closure's claim and its decision; the decision reads the current intent."""
    _work_root(env.root)
    task = _paused_helper(env.root, monkeypatch)
    killed = _close_after_a_dispatch(env, monkeypatch, task)
    real_claim, real_request = ci.claim_intent, ci.request_cancel

    def claim_then_stop(*args, **kwargs):
        claimed = real_claim(*args, **kwargs)
        if claimed and not claimed.get("claim_refused"):
            joined = real_request(env.root, CHILD, reason="the owner pressed Stop", source="http_single", requested_by="owner")
            assert joined["already_requested"] and joined["followup_stop"]
        return claimed

    monkeypatch.setattr(ci, "claim_intent", claim_then_stop)
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: "cancelled"}
    assert killed == [123456789] and load_task_result(env.root, CHILD)["status"] == "cancelled"
    assert CHILD not in ci.active_intents(env.root, strict=True)


def _restart(env):
    """A real restart: the queue is saved, this process's maps are gone, the saved queue is restored."""
    assert env.q.persist_queue_snapshot(reason="restart")
    with env.q._queue_lock:
        env.workers.PENDING.clear()
        env.workers.RUNNING.clear()
        env.workers.WORKERS.clear()
    terminalized = []
    return env.q.restore_pending_from_snapshot(terminalized=terminalized), terminalized


def test_a_restart_between_the_closure_and_its_sweep_closes_the_parked_helper(env):
    """C2: the intent minted, then a restart before the watchdog's pass; the parked row returns and closes."""
    _work_root(env.root)
    _paused_helper(env.root, env.monkeypatch)
    assert close_abandoned_helpers() == [CHILD]
    revived, terminalized = _restart(env)
    assert (revived, terminalized) == (1, []) and _pending(env) is not None
    assert ci.active_intent(env.root, CHILD)["source"] == WORK_FINISHED
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: "cancelled"}
    row = _closed(env)
    assert row["pause_at_close"]["_budget_pause"]["exact_continuation"] is True
    [event] = [row for row in env.events if row.get("type") == "task_done"]
    assert event["task_id"] == CHILD


def test_a_restart_then_an_owner_pause_leaves_the_helper_resumable(env):
    """C2: the restored closure is withdrawn by the owner's Pause; the row stays queued for its Resume."""
    _work_root(env.root)
    _paused_helper(env.root, env.monkeypatch)
    assert close_abandoned_helpers() == [CHILD]
    assert _restart(env)[0] == 1
    _press_pause(env)
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: "not_found"}
    _withdrawn(env)


@pytest.mark.parametrize("consumed", [False, True], ids=["unconsumed", "consumed"])
def test_a_helper_resumed_before_a_restart_is_fenced_like_any_running_row(env, monkeypatch, consumed):
    """C2/I3: an unused grant reparks; a consumed one without saved work gets the ordinary shutdown fence."""
    from ouroboros import budget_pause

    _work_root(env.root)
    task = _paused_helper(env.root, monkeypatch)
    _close_after_a_dispatch(env, monkeypatch, task)
    write_task_result(env.root, CHILD, "running")
    if consumed:
        granted = budget_pause.budget_pause_row(env.root, CHILD)
        budget_pause.set_budget_pause(env.root, CHILD, {
            **granted, "state": budget_pause.STATE_RESUMED,
            "grant": {**granted["grant"], "consumed_at": time.time()}})
    revived, terminalized = _restart(env)
    if not consumed:
        assert revived == 1 and terminalized == []
        assert budget_pause.budget_pause_row(env.root, CHILD)["state"] == budget_pause.STATE_PAUSED
        assert load_task_result(env.root, CHILD)["status"] != "cancelled"
        return
    assert revived == 0 and terminalized == [CHILD]
    intent = ci.active_intent(env.root, CHILD)
    assert intent["source"] == "snapshot_restore" and intent["reason"] == "server_shutdown"
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: "cancelled"}
    row = load_task_result(env.root, CHILD)
    assert row["status"] == "cancelled" and row["cancel_origin"]["source"] == "snapshot_restore"
    assert not ci.active_intent(env.root, CHILD)


def test_a_closure_without_a_row_or_a_worker_settles_through_the_miss_lane(env):
    """C2(c): only a live worker holding the task withdraws the closure; a vanished row is not stranded."""
    _work_root(env.root)
    _paused_helper(env.root, env.monkeypatch)
    assert close_abandoned_helpers() == [CHILD]
    with env.q._queue_lock:
        env.workers.PENDING.clear()  # the row is gone from every map, as a restore that skipped it left it
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: "cancelled"}
    row = load_task_result(env.root, CHILD)
    assert row["status"] == "cancelled" and row["cancel_origin"]["source"] == WORK_FINISHED
    assert not ci.active_intent(env.root, CHILD)


def test_a_withdrawn_closure_leaves_the_helpers_delegated_runs_alone(env, monkeypatch):
    """A1: the irreversible audit (it cancels live delegated runs) runs only after the final recheck."""
    import supervisor.cancel_publication as publication
    from ouroboros import budget_pause

    _work_root(env.root)
    _paused_helper(env.root, env.monkeypatch)
    assert close_abandoned_helpers() == [CHILD]
    audits, real_salvage = [], publication._salvage_cancelled_output

    def salvage_then_grant(q, task, task_id):
        row = budget_pause.budget_pause_row(env.root, CHILD)
        budget_pause.set_budget_pause(env.root, CHILD, {**row, "state": budget_pause.STATE_RESUME_GRANTED})
        return real_salvage(q, task, task_id)

    monkeypatch.setattr(publication, "_salvage_cancelled_output", salvage_then_grant)
    monkeypatch.setattr(publication, "_audit_delegated_runs_on_kill", lambda q, task_id, **kw: audits.append(kw) or {})
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: "not_found"}
    assert audits == []
    _withdrawn(env)


def test_a_stop_joining_while_the_closure_captures_survives_its_withdrawal(env, monkeypatch):
    """C1 at the queued row: the closure is withdrawn by a Resume grant, but the owner's Stop that joined it stands."""
    import supervisor.cancel_publication as publication
    from ouroboros import budget_pause

    _work_root(env.root)
    _paused_helper(env.root, env.monkeypatch)
    assert close_abandoned_helpers() == [CHILD]
    real_salvage = publication._salvage_cancelled_output

    def salvage_while_stop_and_grant(q, task, task_id):
        ci.request_cancel(env.root, CHILD, reason="the owner pressed Stop", source="http_single")
        row = budget_pause.budget_pause_row(env.root, CHILD)
        budget_pause.set_budget_pause(env.root, CHILD, {**row, "state": budget_pause.STATE_RESUME_GRANTED})
        return real_salvage(q, task, task_id)

    monkeypatch.setattr(publication, "_salvage_cancelled_output", salvage_while_stop_and_grant)
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: "failed"}  # given back, never withdrawn
    assert ci.active_intent(env.root, CHILD)["followup_stop"] and _pending(env) is not None
    monkeypatch.setattr(publication, "_salvage_cancelled_output", real_salvage)
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: "cancelled"}
    assert load_task_result(env.root, CHILD)["status"] == "cancelled" and _pending(env) is None
    assert not ci.active_intent(env.root, CHILD)


def test_a_cascade_that_joined_a_closure_cancels_the_queued_helper(env):
    """I1: a cascade's custody of a descendant never runs the closure's rechecks, so it cannot withdraw that cancel."""
    from supervisor.task_lifecycle import cancel_task_custody

    _work_root(env.root)
    _paused_helper(env.root, env.monkeypatch)
    assert close_abandoned_helpers() == [CHILD]
    _press_pause(env)  # the closure alone would now be withdrawn
    ci.request_cancel(env.root, CHILD, reason=f"subtree cancellation of {ROOT}", source="cascade_descendant",
                      requested_by=ROOT, allow_settled_target=True)
    assert cancel_task_custody(CHILD, deliver=False, cascade_claim={"task_id": ROOT, "scope": "cascade"}) == "cancelled"
    assert load_task_result(env.root, CHILD)["status"] == "cancelled" and _pending(env) is None


class _Engine:
    """The engine's attach-only transport; every run reads as running and each cancel is recorded."""

    def __init__(self, cancelled):
        self.cancelled = cancelled

    def handshake(self):
        return {}

    def get_run(self, run_id):
        return {"summary": {"runId": run_id, "state": "running"}}

    def cancel_run(self, run_id, reason=""):
        from ouroboros.delegate_custody_current import run

        owner = getattr(run(_Engine.root, run_id), "task_id", "") or CHILD
        # The status of the run's own task at the moment the cancel was sent.
        self.cancelled.append((run_id, reason, (load_task_result(_Engine.root, owner) or {}).get("status")))
        return {"accepted": True, "status": "requested"}

    def close(self):
        pass


def _engine(env, monkeypatch):
    """The real reconcile runs over the attach-only transport above; returns the cancels it sends."""
    import ouroboros.delegate_custody_reconcile as reconcile

    cancelled = []
    _Engine.root = env.root
    monkeypatch.setattr(reconcile, "attach_only_gateway", lambda: _Engine(cancelled))
    return cancelled


def _started_reviewer(env, monkeypatch, task_id=CHILD):
    """A review panel's run the helper started."""
    from ouroboros import delegate_custody as custody

    cancelled = _engine(env, monkeypatch)
    assert custody.record_started(env.root, custody.RunCustody(
        run_id=f"run-{task_id}", task_id=task_id, route_id="codex", model="probe",
        invocation_id=f"review-{task_id}", source="review_substrate", access="readonly"))
    return cancelled


@pytest.mark.parametrize("moment", ["before_write", "inside_write"])
def test_an_authority_parked_helpers_reviewer_survives_a_pause_that_wins_the_last_transaction(env, monkeypatch, moment):
    """#3047 Astra r3: the real authority park keeps the started reviewer; the owner's Pause then wins the closure's final
    write, so the helper is retained and nothing irreversible may have been sent to its reviewer."""
    from ouroboros import budget_pause as bp
    from supervisor.events_budget import install_exact_budget_pause
    from tests._budget_pause_exact_helpers import _fast_hold, _loop_ctx

    _work_root(env.root)
    cancelled = _started_reviewer(env, monkeypatch)
    task = {"id": CHILD, "type": "task", "chat_id": 7, "root_task_id": ROOT, "parent_task_id": ROOT,
            "delegation_role": "subagent", "_attempt": 1}
    write_task_result(env.root, CHILD, "running", root_task_id=ROOT, parent_task_id=ROOT, delegation_role="subagent", chat_id=7)
    ctx, limit = _loop_ctx(env.root, CHILD)
    ctx.root_task_id = ROOT
    ctx.task_metadata = {"root_task_id": ROOT, "parent_task_id": ROOT, "delegation_role": "subagent"}
    env.workers.RUNNING[CHILD] = {"task": task, "worker_id": 0, "attempt": 1}
    env.workers.WORKERS[0] = SimpleNamespace(busy_task_id=CHILD)
    _fast_hold(monkeypatch, bp)
    with pytest.raises(bp.BudgetPauseRequested) as parked:
        bp.request_pause(limit, rail=bp.RAIL_CONTROL_AUTHORITY, scope="task",
                         reason_text="control authority temporarily unreadable", root_task_id=ROOT)
    row = parked.value.pause
    assert row["reason"] == "authority" and row["external_runs"]["runs"][0]["review_owned"] and cancelled == []
    install_exact_budget_pause(_supervisor_ctx(env.root, env.workers, env.q, [], []), CHILD,
                               bp.exact_pause_marker(row)["checkpoint"], evt=bp.pause_event(task, row))
    bp.end_dispatch_fence(CHILD)
    _paused_helper(env.root, monkeypatch, task_id=SIBLING)
    assert set(close_abandoned_helpers()) == {CHILD, SIBLING}
    fired = _fire_before_write(env, monkeypatch, moment, lambda: _press_pause(env))
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: "not_found", SIBLING: "not_found"}
    assert fired
    _withdrawn(env)
    _withdrawn(env, SIBLING)
    assert cancelled == []


def test_a_closure_that_wins_its_write_then_stops_the_helpers_reviewer(env, monkeypatch):
    """The other side: the owner's choice closes the helper, and its run is stopped behind the committed cancellation."""
    _work_root(env.root)
    _paused_helper(env.root, env.monkeypatch)
    cancelled = _started_reviewer(env, monkeypatch)
    assert _sweep(env) == ([CHILD], {CHILD: "cancelled"})
    row = _closed(env)
    assert cancelled == [(f"run-{CHILD}", "owner_task_gone", "cancelled")]  # sent only once the closure was durable
    assert not row.get("delegated_runs_unreconciled")  # stopped, so nothing is left open to disclose


def _die_after_the_closure_write(monkeypatch):
    """The process ends right after the committed cancelled write, before the runs are stopped or the intent settled."""
    import supervisor.ended_work_helpers as helpers

    class Died(BaseException):
        pass

    def die(q, task_id):
        assert load_task_result(q.DRIVE_ROOT, task_id)["status"] == "cancelled"
        raise Died()

    monkeypatch.setattr(helpers, "stop_closed_runs", die)
    return Died


@pytest.mark.parametrize("saved_row", [True, False], ids=["snapshot_keeps_row", "snapshot_lost_row"])
def test_a_restart_after_the_closure_write_still_stops_the_runs_and_settles_once(env, monkeypatch, saved_row):
    """Interruption/replay: the open intent survives the death, and the settled lane reconciles behind the durable closure."""
    import ouroboros.platform_layer as platform
    import supervisor.ended_work_helpers as helpers

    _work_root(env.root)
    _paused_helper(env.root, env.monkeypatch)
    cancelled = _started_reviewer(env, monkeypatch)
    assert close_abandoned_helpers() == [CHILD]
    if saved_row:
        assert env.q.persist_queue_snapshot(reason="before the death")
    real_stop = helpers.stop_closed_runs
    with pytest.raises(_die_after_the_closure_write(monkeypatch)):
        sweep_cancel_intents(now=time.time() + 3600)
    assert cancelled == [] and ci.active_intent(env.root, CHILD)["source"] == WORK_FINISHED
    monkeypatch.setattr(helpers, "stop_closed_runs", real_stop)
    monkeypatch.setattr(platform, "pid_is_alive", lambda pid: False)  # the process is gone: its claim and maps with it
    with env.q._queue_lock:
        env.workers.PENDING.clear()
        env.workers.RUNNING.clear()
        env.workers.WORKERS.clear()
    env.q.restore_pending_from_snapshot(terminalized=[])
    assert sweep_cancel_intents(now=time.time() + 7200) == {CHILD: "already_settled"}
    row = _closed(env)
    assert [run for run, _reason, _status in cancelled] == [f"run-{CHILD}"]
    assert row["result"].startswith("Closed:")
    assert sweep_cancel_intents(now=time.time() + 10800) == {}  # nothing is replayed twice


# #3047 R7: delegated work of anyone but the abandoned helpers is the work still going on.


def _run_in_work(env, owner):
    """An ordinary delegated run ``owner`` started in the work, through custody's own writer."""
    from ouroboros import delegate_custody as custody

    assert custody.record_started(env.root, custody.RunCustody(
        run_id=f"run-{owner}", task_id=owner, root_task_id=ROOT, route_id="codex", model="probe", access="readonly"))
    return f"run-{owner}"


def _member_of_the_work(env, owner):
    """The ended work and the member of it named: the root, its finished retry, a finished helper or a kept one."""
    from tests._cancel_intents_shared import _write_root_retry_pair

    if owner == "work-retry":
        _write_root_retry_pair(env.root, ROOT, owner, new_status="completed")
        return
    _work_root(env.root)
    if owner != ROOT:
        write_task_result(env.root, owner, "failed" if owner == "finished-helper" else "scheduled",
                          root_task_id=ROOT, parent_task_id=ROOT, delegation_role="subagent")
    if owner == "kept-helper":  # the owner holds it: queued, not live, and never closed
        env.workers.PENDING.append({"id": owner, "type": "task", "chat_id": 7, "root_task_id": ROOT, "parent_task_id": ROOT,
                                    "_attempt": 1, "delegation_role": "subagent", "_owner_hold": {"reason": "owner"}})


@pytest.mark.parametrize("owner", [ROOT, "work-retry", "finished-helper", "kept-helper"])
def test_a_run_of_anyone_else_in_the_work_keeps_the_helper_until_it_settles(env, owner):
    from ouroboros import delegate_custody as custody

    _member_of_the_work(env, owner)
    _paused_helper(env.root, env.monkeypatch)
    run_id = _run_in_work(env, owner)
    assert _sweep(env) == ([], {})
    _retained(env)
    assert custody.emit(env.root, custody.SETTLED, {"run_id": run_id, "task_id": owner, "state": "succeeded"})
    assert _sweep(env)[0] == [CHILD]
    _closed(env)


def test_a_run_its_engine_reported_finished_keeps_no_helper(env):
    """Its settlement is still owed, but it runs nothing: the work projection counts no member for it either."""
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_reconcile import tree_open_runs

    _work_root(env.root)
    _paused_helper(env.root, env.monkeypatch)
    run_id = _run_in_work(env, ROOT)
    assert custody.emit(env.root, custody.TERMINAL_OBSERVED, {"run_id": run_id, "task_id": ROOT, "state": "succeeded",
                                                               "observation_seq": 1, "source_ref": {"sha256": "0" * 64}})
    [entry] = tree_open_runs(env.root, ROOT)  # still unsettled: the cascade Stop would reach it
    assert entry.terminal_state == "succeeded"
    assert _sweep(env)[0] == [CHILD]
    _closed(env)


@pytest.mark.parametrize("owner", [ROOT, "finished-helper"])
def test_a_start_of_unknown_outcome_in_the_work_keeps_the_helper_until_it_is_answered(env, owner):
    """The engine may have accepted it while no run is bound yet; a definite refusal answers it."""
    from ouroboros import delegate_custody as custody

    _member_of_the_work(env, owner)
    _paused_helper(env.root, env.monkeypatch)
    assert custody.record_start_requested(env.root, task_id=owner, root_task_id=ROOT, invocation_id="inv-unknown",
                                          idempotency_key="inv-unknown", route="test", request={"prompt": "work"})
    assert _sweep(env) == ([], {})
    _retained(env)
    assert custody.emit(env.root, custody.START_FAILED, {"invocation_id": "inv-unknown", "task_id": owner, "definite": True})
    assert _sweep(env)[0] == [CHILD]
    _closed(env)


@pytest.mark.parametrize("read", ["pending", "open_held_runs"])
def test_unreadable_custody_keeps_the_helper_at_discovery_and_at_its_recheck(env, monkeypatch, read):
    import ouroboros.delegate_custody_current as current

    _work_root(env.root)
    _paused_helper(env.root, env.monkeypatch)
    real = getattr(current, read)

    def unreadable(*_args, **_kwargs):
        raise OSError("custody unreadable")

    monkeypatch.setattr(current, read, unreadable)
    assert _sweep(env) == ([], {})
    _retained(env)
    monkeypatch.setattr(current, read, real)
    assert close_abandoned_helpers() == [CHILD]
    monkeypatch.setattr(current, read, unreadable)
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: "not_found"}
    _withdrawn(env)


def test_a_run_of_the_work_recorded_after_discovery_withdraws_the_closure(env):
    """The recheck holds discovery's whole verdict, the work's delegated runs included."""
    _work_root(env.root)
    _paused_helper(env.root, env.monkeypatch)
    assert close_abandoned_helpers() == [CHILD]
    _run_in_work(env, ROOT)
    assert sweep_cancel_intents(now=time.time() + 3600) == {CHILD: "not_found"}
    _withdrawn(env)


def test_the_helpers_own_run_in_the_work_keeps_nothing_and_its_closure_stops_it(env, monkeypatch):
    _work_root(env.root)
    _paused_helper(env.root, env.monkeypatch)
    cancelled = _engine(env, monkeypatch)
    run_id = _run_in_work(env, CHILD)
    assert _sweep(env) == ([CHILD], {CHILD: "cancelled"})
    _closed(env)
    assert cancelled == [(run_id, "owner_task_gone", "cancelled")]  # sent once its closure was durable


def test_abandoned_helpers_own_runs_never_hold_one_another(env, monkeypatch):
    """Each closure stops its own helper's runs, which read open until the engine obeys: neither a helper closed in the
    same pass nor one closed earlier keeps another, so they close together."""
    from ouroboros.delegate_custody_reconcile import tree_open_runs

    _work_root(env.root)
    cancelled = _engine(env, monkeypatch)
    for task_id in (CHILD, SIBLING):
        _paused_helper(env.root, env.monkeypatch, task_id=task_id)
        _run_in_work(env, task_id)
    minted, outcomes = _sweep(env)
    assert sorted(minted) == sorted([CHILD, SIBLING]) and outcomes == {CHILD: "cancelled", SIBLING: "cancelled"}
    late = "work-child-3"  # parked after those closed, their runs still reading open
    _paused_helper(env.root, env.monkeypatch, task_id=late)
    _run_in_work(env, late)
    assert _sweep(env) == ([late], {late: "cancelled"})
    helpers = (CHILD, SIBLING, late)
    for task_id in helpers:
        _closed(env, task_id)
    assert sorted(cancelled) == sorted((f"run-{task_id}", "owner_task_gone", "cancelled") for task_id in helpers)
    assert {entry.run_id for entry in tree_open_runs(env.root, ROOT)} == {f"run-{task_id}" for task_id in helpers}


def test_one_custody_read_serves_every_ended_work_with_a_parked_helper(env, monkeypatch):
    import ouroboros.delegate_custody_current as current

    reads = []
    for name in ("pending", "open_held_runs"):
        monkeypatch.setattr(current, name, lambda root, _real=getattr(current, name), _name=name: (
            reads.append(_name), _real(root))[1])
    _work_root(env.root, task_id="work-root-2", status="running")
    _paused_helper(env.root, monkeypatch, task_id="work-2-child", work="work-root-2")
    _member_of_the_work(env, "kept-helper")  # ended, and its one helper is the owner's to keep
    assert close_abandoned_helpers() == [] and reads == []  # no closure to ask: custody is not read
    _paused_helper(env.root, monkeypatch)
    _paused_helper(env.root, monkeypatch, task_id=SIBLING)
    write_task_result(env.root, "work-root-2", "completed", result="the answer")
    assert sorted(close_abandoned_helpers()) == sorted([CHILD, SIBLING, "work-2-child"])
    assert sorted(reads) == ["open_held_runs", "pending"]
