"""#2380/#2411: the Resume door's latest answer stays on the task's own record, bound to the control it answered about.

The real HTTP Continue, Pause and Resume handlers, the one Resume door
(``queue_transitions.resume_budget_paused_task``), the mind and child doors, the task store, the census and the
server's work projection run here. The door's whole answer is kept on the task's record (but the held calls an owner
may go on past), with when it was asked and answered, who asked (bound by the door it came through) and the control
it answered about (``work_stop.control_identity``, read before the door acted). The work projection says a refusal
beside the current stop only while that stop is the same control, never as its reason, a door or a turn: a newer
stop, even one landing while the door still worked or in the same words after a release, never takes an older answer
(each records its occurrence, ``work_stop._identity_facts``). Nothing the door decides or returns
changes: a failed write is not a refusal, an unknown id creates no task, an answer asked before a newer attempt never
replaces it, and the census reads the attempt from the body facts it memoizes by row, writing nothing.
The refusals here come from a fact no wait settles (a delegated start whose outcome is unknown beside a running
leaf); a Resume over writers that all run returns the work to its writer wait instead
(``tests/test_continue_resume_writer_wait.py``).
"""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from unittest import mock

import pytest

from ouroboros import delegate_custody as dc
from tests._work_projection_shared import state_fields, work_fields
from tests.test_continue_blocker_diagnostics import _held, _http
from tests.test_continue_independent_review import PRED, _current_pool  # noqa: F401 - pytest fixture
from tests.test_owner_continue import NONCE


LEAF = "delegated run run-leaf is running"
UNKNOWN_START = "a delegated start has an unknown outcome (no run was recorded)"
FACTS = f"{LEAF}; {UNKNOWN_START}"


def _stored(root, task_id):
    from ouroboros.task_results import load_task_result

    return (load_task_result(root, task_id) or {}).get("resume_attempt")


def _events(root, kind):
    path = root / "logs" / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return [row for row in rows if row.get("type") == kind]


def _work(root, episode, *, attempts=True):
    """The server's work for ``episode`` from the real census; ``attempts=False`` reads no stored attempt."""
    from ouroboros.gateway import work_projection as wp

    with mock.patch.object(wp._Observations, "resume_attempt", wp._Observations.resume_attempt if attempts
                           else lambda self, task_id, drive_root=None: {}):
        works = state_fields(root)["works"]
    [work] = [work for work in works if work["episode"]["task_id"] == episode]
    return work


def _without_attempt(work):
    """The work as it would be without its dated attempt: everything else the projection says."""
    states = {task: {key: value for key, value in state.items() if key != "last_resume"}
              for task, state in work["members"]["states"].items()}
    return {**{key: value for key, value in work.items() if key != "last_resume"},
            "members": {**work["members"], "states": states}}


def _paused_held_continue(tmp_path, monkeypatch, blocker="running_and_unknown_start"):
    """#2411's shape: a Continue held for its predecessor's writers, then paused by the owner. By default its leaf
    runs beside a start whose outcome is unknown, which no wait settles: the owner's Resume is refused."""
    from ouroboros.gateway.task_continue import api_task_continue
    from ouroboros.gateway.task_pause import api_task_pause

    queue, workers = _held(tmp_path, monkeypatch, blocker)
    status, ack = _http(api_task_continue, PRED, {"action_nonce": NONCE})
    assert status == 200 and ack["held"]
    successor = ack["successor_task_id"]
    # The owner's own client declares its press (``api_client.pauseTask``): the Pause names its author.
    status, pause = _http(api_task_pause, successor, {"request_id": "pause-held-continue", "actor": "owner"})
    assert status in {200, 202} and pause["ok"]
    return queue, workers, successor


def _settle_leaf(root):
    assert dc.emit(root, dc.SETTLED, {"run_id": "run-leaf", "task_id": PRED, "route": "codex",
                                      "model": "leaf-model", "state": "completed"})
    dc._CUSTODY.clear()


def _resolve_unknown_start(root):
    """The unbound start fails definitively: its outcome is known, only the running leaf remains."""
    assert dc.emit(root, dc.START_FAILED, {"invocation_id": "inv-unbound", "task_id": PRED, "definite": True,
                                           "error": "refused"})
    dc._CUSTODY.clear()


def test_a_refused_owner_resume_is_kept_and_said_beside_the_pause_until_a_newer_attempt(tmp_path, monkeypatch):
    from ouroboros import owner_pause
    from ouroboros.gateway.tasks import api_task_resume
    from ouroboros.work_stop import _instant, resume_asked

    queue, workers, successor = _paused_held_continue(tmp_path, monkeypatch)
    before = _work(tmp_path, successor)
    assert (before["activity"], before["because"][0]["code"]) == ("paused", "paused_by_owner")
    assert "last_resume" not in before and _stored(tmp_path, successor) is None

    status, refusal = _http(api_task_resume, successor)
    assert status == 409 and refusal["error"] == "predecessor_writers_unsettled", refusal
    stored = _stored(tmp_path, successor)
    # The door's whole typed answer in its words, when it was asked and answered, that the owner's own door asked,
    # and the control it answered about: the owner's Pause, by that fence's own records.
    fence = owner_pause.read_fence(tmp_path, successor)
    assert stored["ok"] is False and stored["by"] == {"actor": "owner"} and "selected_by" not in stored
    assert stored["answer"]["ok"] is False and "acceptable_unknown" not in stored["answer"]
    assert (stored["answer"]["error"], stored["answer"]["message"]) == (refusal["error"], refusal["message"]) == (
        "predecessor_writers_unsettled", f"Resume refused: {FACTS}")
    assert stored["answer"]["blockers"] == refusal["blockers"] and len(refusal["blockers"]) == 2
    assert (stored["control"]["cause"], stored["control"]["scope"], stored["control"]["fence_id"]) == (
        "pause", "tree", fence["fence_id"])
    assert _instant(stored["at"]) <= _instant(stored["answered_at"])
    # The original answer stays whole in the event history too, beside the same control.
    [event] = _events(tmp_path, "task_resume_outcome")
    assert (event["task_id"], event["actor"], event["outcome"]["error"], event["control"]) == (
        successor, "owner", refusal["error"], stored["control"])

    work = _work(tmp_path, successor)
    said = {"at": stored["at"], "by": {"actor": "owner"}, "error": "predecessor_writers_unsettled", "text": FACTS}
    assert work["last_resume"] == said and work["members"]["states"][successor]["last_resume"] == said
    # Dated history beside the Pause: everything else the work says and offers is what it says without it, and the
    # Pause, its reason and its Resume door are what they were before the press.
    assert _without_attempt(work) == _work(tmp_path, successor, attempts=False)
    assert (work["activity"], work["because"], work["turn"]) == (before["activity"], before["because"], before["turn"])
    assert {"action": "resume", "target": successor} in work["controls"]

    # The unknown start resolves; the leaf still runs: the dated refusal stays as said beside the same Pause.
    _resolve_unknown_start(tmp_path)
    assert _work(tmp_path, successor)["last_resume"] == said
    # An attempt asked now but answered only after the next one (a slow door).
    late = resume_asked(queue, successor)
    # A newer attempt: every writer left runs, so the work returns to its writer wait and says no refusal.
    status, accepted = _http(api_task_resume, successor)
    assert status == 200 and accepted["ok"] and accepted["held"], accepted
    kept = _stored(tmp_path, successor)
    assert kept["ok"] is True and _instant(kept["at"]) > _instant(stored["at"]) and kept["answer"]["held"] is True
    waiting = _work(tmp_path, successor)
    assert "last_resume" not in waiting and "resume" not in {control["action"] for control in waiting["controls"]}
    # The slower, older attempt's refusal lands last and never replaces the newer answer.
    refused = {"ok": False, "error": "selection_authority_changed", "message": "Resume refused: selection authority changed"}
    assert late(refused) is refused
    assert _stored(tmp_path, successor) == kept


def _restart_held(tmp_path, monkeypatch, workers, queue, door="owner_restart"):
    """The owner's Restart over a held Continue: the restart's own hold replaces its writer wait (Resume releases it)."""
    from ouroboros import delegate_recovery
    from supervisor import active_activity, queue_snapshot

    registry = active_activity.DirectActivityRegistry()
    monkeypatch.setattr(active_activity, "get_direct_activity_registry", lambda: registry)
    monkeypatch.setattr(queue_snapshot, "_RESTORED", set())
    monkeypatch.setattr(queue_snapshot, "_RESTORING", {})
    if door == "owner_restart":
        delegate_recovery.prepare_owner_restart(tmp_path, legacy=True)
    workers.kill_workers(force=True, terminal_status="cancelled", result_reason="Server shutdown.",
                         stop_source="server_shutdown", retain_saved_work=True, preserve_pending=True)
    workers.PENDING[:] = []
    queue.init(tmp_path)
    assert queue.restore_pending_from_snapshot() >= 1


def test_a_newer_pause_landing_while_a_refused_resume_works_is_never_said_beside_it(tmp_path, monkeypatch):
    """The owner's Resume of a Continue the Restart holds observes its predecessor's writers off the queue lock; the
    owner's Pause lands while it does. The door then refuses (the authority it observed changed), answering about
    the Restart's hold: the newer Pause stands, and that refusal is never said beside it, although it was answered
    after the Pause was pressed (the reading by time the record replaced would have said it)."""
    from ouroboros.gateway.task_continue import api_task_continue
    from ouroboros.gateway.task_pause import api_task_pause
    from ouroboros.gateway.tasks import api_task_resume
    from ouroboros.work_stop import _instant, hold_fact
    from supervisor import events_budget

    queue, workers = _held(tmp_path, monkeypatch, "running_and_unknown_start")
    status, ack = _http(api_task_continue, PRED, {"action_nonce": NONCE})
    assert status == 200 and ack["held"]
    successor = ack["successor_task_id"]
    _restart_held(tmp_path, monkeypatch, workers, queue)
    row = next(item for item in workers.PENDING if item["id"] == successor)
    assert hold_fact(row)["reason"] == "owner_restart_hold"
    held = _work(tmp_path, successor)
    assert (held["activity"], held["because"][0]["code"]) == ("paused", "after_restart") and "last_resume" not in held

    observe = events_budget.observe_held_budget_selection
    pressed = {}

    def observe_while_the_owner_pauses(q, task_id):
        observation = observe(q, task_id)
        if task_id == successor and not pressed:  # the Pause door runs on its own thread, as a second request would
            thread = threading.Thread(target=lambda: pressed.update(answer=_http(
                api_task_pause, successor, {"request_id": "pause-during-resume", "actor": "owner"})))
            thread.start()
            thread.join(timeout=60)
        return observation

    monkeypatch.setattr(events_budget, "observe_held_budget_selection", observe_while_the_owner_pauses)
    status, refusal = _http(api_task_resume, successor)
    assert pressed["answer"][0] in {200, 202} and pressed["answer"][1]["ok"], pressed
    assert status == 409 and refusal["error"] == "selection_authority_changed", refusal
    stored = _stored(tmp_path, successor)
    assert stored["ok"] is False and stored["control"]["reason"] == "owner_restart_hold", stored
    work = _work(tmp_path, successor)
    assert (work["activity"], work["because"][0]["code"]) == ("paused", "paused_by_owner")
    assert "last_resume" not in work and "last_resume" not in work["members"]["states"][successor]
    assert {"action": "resume", "target": successor} in work["controls"]
    # Answered after the newer Pause's own press: a time comparison would have attached it to that Pause.
    assert _instant(stored["answered_at"]) >= _instant(work["because"][0]["since"])
    # The same answer stays the record's history, bound to the Restart's hold it answered about.
    [event] = _events(tmp_path, "task_resume_outcome")
    assert event["outcome"]["error"] == "selection_authority_changed" and event["control"] == stored["control"]


def _attempts_unwritable(monkeypatch):
    """The store refuses, inside its transaction, exactly the writes that would keep an attempt; undo with the context."""
    from ouroboros import task_store
    from ouroboros.work_stop import ATTEMPT_KEY

    real = task_store.patch_document

    def unwritable(root, task_id, mutator):
        def guarded(current):
            updated = mutator(current)
            if updated is not None and updated.get(ATTEMPT_KEY) != current.get(ATTEMPT_KEY):
                raise OSError("store unavailable")
            return updated
        return real(root, task_id, guarded)

    monkeypatch.setattr(task_store, "patch_document", unwritable)


def test_a_standing_restart_hold_keeps_its_refusal_and_a_stop_declared_anew_takes_none(tmp_path, monkeypatch):
    """Two restarts with nothing between them are one hold (the door matrix keeps it, and its refusal stays said). Two
    with a release between them are two stops in the same words: restart A's refusal, the outcome it named long
    resolved, is never said beside restart B, even when the Resume that ended A could not keep its own answer, which
    never gates that Resume."""
    from ouroboros.gateway.task_continue import api_task_continue
    from ouroboros.gateway.tasks import api_task_resume
    from ouroboros.work_stop import hold_fact

    queue, workers = _held(tmp_path, monkeypatch, "running_and_unknown_start")
    status, ack = _http(api_task_continue, PRED, {"action_nonce": NONCE})
    assert status == 200 and ack["held"]
    successor = ack["successor_task_id"]

    def held():
        return hold_fact(next(row for row in workers.PENDING if row["id"] == successor))

    _restart_held(tmp_path, monkeypatch, workers, queue)
    restart_a = dict(held())
    status, refusal = _http(api_task_resume, successor)
    assert status == 409 and refusal["error"] == "predecessor_writers_unsettled", refusal
    refused = _stored(tmp_path, successor)
    assert refused["control"]["held_at"] == restart_a["held_at"] and UNKNOWN_START in refused["answer"]["message"]
    said = _work(tmp_path, successor)["last_resume"]
    assert UNKNOWN_START in said["text"]

    _restart_held(tmp_path, monkeypatch, workers, queue)
    assert held() == restart_a, "a second restart keeps the standing hold"
    assert _work(tmp_path, successor)["last_resume"] == said, "the same stop keeps its dated refusal"

    _resolve_unknown_start(tmp_path)
    with monkeypatch.context() as patched:
        _attempts_unwritable(patched)
        status, accepted = _http(api_task_resume, successor)
    assert status == 200 and accepted["ok"] and accepted["held"], "an answer that cannot be kept gates nothing"
    assert _stored(tmp_path, successor) == refused, "the refusal is still the record's latest attempt"
    assert "last_resume" not in _work(tmp_path, successor), "nor is it said beside the wait that replaced its stop"

    _restart_held(tmp_path, monkeypatch, workers, queue)
    restart_b = dict(held())
    assert restart_b["reason"] == restart_a["reason"] and restart_b["held_at"] != restart_a["held_at"]
    assert _stored(tmp_path, successor) == refused
    work = _work(tmp_path, successor)
    assert (work["activity"], work["because"][0]["code"]) == ("paused", "after_restart")
    assert "last_resume" not in work and "last_resume" not in work["members"]["states"][successor]
    assert UNKNOWN_START not in json.dumps(work), "the outcome resolved under the earlier stop is not said again"
    assert {"action": "resume", "target": successor} in work["controls"]
    # What the new stop does get asked is its own: an answer about it is said beside it.
    late = _refused_about_now(queue, successor)
    assert late["control"]["held_at"] == restart_b["held_at"]
    assert _work(tmp_path, successor)["last_resume"]["error"] == "selection_authority_changed"


def _refused_about_now(queue, task_id):
    """A door's refusal about whatever holds the task now, through the one keeper of its answer."""
    from ouroboros.control_actor import owner_door
    from ouroboros.work_stop import resume_asked

    with owner_door("api_task_resume"):
        answered = resume_asked(queue, task_id)
    answered({"ok": False, "error": "selection_authority_changed", "message": "Resume refused: selection authority changed"})
    return _stored(queue.DRIVE_ROOT, task_id)


def test_a_later_money_stop_in_the_same_words_takes_no_refusal_of_the_one_before(tmp_path, monkeypatch):
    """A member's money stop names no fence, pause or hold: when it paused is its occurrence. Refused, released by a
    Resume whose answer could not be kept, then stopped by money again: that newer stop says nothing of the older."""
    from ouroboros import usage_accounting
    from ouroboros.gateway.tasks import api_task_resume
    from ouroboros.task_results import STATUS_SCHEDULED, write_task_result
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_budget_pause_holds import _idle_worker

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    money = {"left": 0.0}
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: money["left"])
    row = {"id": "m1", "type": "task", "chat_id": 7, "root_task_id": "m1", "_attempt": 1, "admitted_dispatch": "none"}
    workers.PENDING.append(row)
    write_task_result(tmp_path, "m1", STATUS_SCHEDULED, root_task_id="m1", chat_id=7)
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    first = row["_budget_pause"]
    assert first["status"] == "paused_before_dispatch" and first["scope"] == "global" and sent == []
    real = usage_accounting.usage_breakdown

    def unreadable(root, *args, task_id=None, **kwargs):
        if task_id == "m1":
            raise OSError("usage ledger unreadable")
        return real(root, *args, task_id=task_id, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(usage_accounting, "usage_breakdown", unreadable)
        status, refusal = _http(api_task_resume, "m1")
    assert status == 409 and refusal["error"] == "accounting_unavailable", refusal
    refused = _stored(tmp_path, "m1")
    assert refused["control"]["paused_at"] == first["paused_at"]
    assert _work(tmp_path, "m1")["last_resume"]["error"] == "accounting_unavailable", "the same stop says it"

    money["left"] = 5.0
    with monkeypatch.context() as patched:
        _attempts_unwritable(patched)
        status, accepted = _http(api_task_resume, "m1")
    assert status == 200 and accepted["ok"], accepted
    money["left"] = 0.0
    workers.assign_tasks()
    assert sent == [] and row["_budget_pause"]["paused_at"] != first["paused_at"], "stopped by money again"
    assert _stored(tmp_path, "m1") == refused
    work = _work(tmp_path, "m1")
    assert (work["activity"], work["because"][0]["code"]) == ("paused", "spending_limit")
    assert "last_resume" not in work and "last_resume" not in work["members"]["states"]["m1"]


def test_an_exact_pause_keeps_its_own_refusal_by_its_pause_id(tmp_path, monkeypatch):
    """The door and the census read an exact pause alike (its durable pause row names it), so its refusal is said beside
    it, and a later pause of the same task and words, with its own id, says nothing of it."""
    from ouroboros.budget_pause import budget_pause_row, set_budget_pause
    from ouroboros.gateway.tasks import api_task_resume
    from tests._budget_pause_exact_helpers import _install_queue, _parked

    _queue, state, _workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 0.0)
    _task, row = _parked(tmp_path, monkeypatch, task_id="exact-1")
    status, refusal = _http(api_task_resume, "exact-1")
    assert status == 409 and refusal["error"] == "budget_still_exhausted", refusal
    refused = _stored(tmp_path, "exact-1")
    assert refused["control"]["pause_id"] == row["pause_id"]
    work = _work(tmp_path, "exact-1")
    assert (work["activity"], work["because"][0]["code"]) == ("paused", "spending_limit")
    assert work["last_resume"]["error"] == "budget_still_exhausted"
    # A newer pause of this task: the same words, its own durable id.
    durable = budget_pause_row(tmp_path, "exact-1")
    set_budget_pause(tmp_path, "exact-1", {**durable, "pause_id": "newer-pause"},
                     expected_pause_id=durable["pause_id"], expected_state=durable["state"])
    newer = _work(tmp_path, "exact-1")
    assert newer["because"][0]["code"] == "spending_limit" and "last_resume" not in newer


def test_a_member_awaiting_selection_is_a_new_stop_after_each_lifted_pause():
    from ouroboros.work_stop import control_identity, stop_presentation

    def awaiting(fence_id):
        return control_identity(stop_presentation({"id": "m", "root_task_id": "r"}, {
            "owner": {"awaiting_selection": True, "state": "paused", "fence_id": fence_id}}))

    assert awaiting("f1")["cause"] == "awaiting_selection" and awaiting("f1") == awaiting("f1")
    assert awaiting("f1") != awaiting("f2")


def test_an_owner_fence_settling_is_still_the_same_stop():
    """A settling fence restamps its ``paused_at``: a refusal about it, read before it settled, stays said."""
    from ouroboros.work_stop import control_identity, stop_presentation

    def fence(**state):
        return control_identity(stop_presentation({"id": "r"}, {"owner": {
            "fence_id": "f1", "requested_at": "2026-10-17T15:00:00Z", **state}}))

    assert fence(state="requested") == fence(state="paused", paused_at="2026-10-17T15:02:00Z")


def test_a_wait_declared_anew_takes_no_start_refusal_of_the_wait_before_it():
    """The host's refused start of an earlier wait is that wait's fact: a wait in the same words declared later (a
    Resume returned the Continue to it) says what it waits for, never the older refusal as its reason."""
    wait = {"reason": "continuation_writer_unsettled", "release": "none", "held_at": "2026-10-17T15:10:00Z",
            "blockers": [], "predecessor_task_id": "ps"}
    control = {"cause": "wait", "ui_code": "continuation_writer_unsettled", "reason": "continuation_writer_unsettled",
               "release": "none"}
    row = {"activity_id": "cs", "phase": "budget_paused", "kind": "managed_task", "chat_id": 42, "task_attempt": 1,
           "started_at": 100.0, "pause_cause": "unknown", "can_resume": False}
    raw = {"cs": {"_budget_pause_hold": wait}}

    def said(held_at):
        attempt = _refused({**control, "held_at": held_at}, by={"actor": "host"}, selected_by="reconciliation")
        return work_fields([row], raw, attempts={"cs": attempt})["works"][0]

    this_wait = said(wait["held_at"])
    assert this_wait["because"][0]["code"] == "start_refused" and this_wait["last_resume"]["by"] == {"actor": "host"}
    earlier = said("2026-10-17T15:00:00Z")
    assert earlier["because"][0]["code"] == "earlier_work_unconfirmed" and "last_resume" not in earlier


def test_a_failed_write_changes_nothing_the_door_returned_and_repeats_nothing(tmp_path, monkeypatch):
    from ouroboros.gateway.tasks import api_task_resume
    from ouroboros.work_stop import resume_outcome
    from supervisor import queue_transitions

    queue, workers, successor = _paused_held_continue(tmp_path, monkeypatch)
    answers = []
    door = queue_transitions._resume_budget_paused_task
    monkeypatch.setattr(queue_transitions, "_resume_budget_paused_task",
                        lambda *a, **k: answers.append(door(*a, **k)) or answers[-1])
    _attempts_unwritable(monkeypatch)
    status, refusal = _http(api_task_resume, successor)
    assert status == 409 and len(answers) == 1, "the door ran once"
    formatted = resume_outcome(answers[0])
    assert {key: refusal[key] for key in ("error", "blockers", "message")} == {
        key: formatted[key] for key in ("error", "blockers", "message")}
    assert _stored(tmp_path, successor) is None
    # The owner's answer still reaches the event history whole: the record was the part that failed.
    [event] = _events(tmp_path, "task_resume_outcome")
    assert event["outcome"]["error"] == refusal["error"]

    _resolve_unknown_start(tmp_path)
    status, accepted = _http(api_task_resume, successor)
    assert status == 200 and accepted["ok"] and accepted["held"] and len(answers) == 2, "an unkept answer is the answer"
    hold = next(row for row in workers.PENDING if row["id"] == successor).get("_budget_pause_hold")
    assert hold["reason"] == "continuation_writer_unsettled" and _stored(tmp_path, successor) is None


def test_an_unknown_task_gets_its_refusal_and_no_record(tmp_path, monkeypatch):
    from ouroboros import task_store
    from ouroboros.gateway.tasks import api_task_resume
    from tests._budget_pause_exact_helpers import _install_queue

    _install_queue(tmp_path, monkeypatch)
    status, refusal = _http(api_task_resume, "ghost-task-1")
    assert status == 404 and refusal["error"] == "task_not_pending", refusal
    assert not task_store.exists(tmp_path, "ghost-task-1"), "no phantom task for an unknown id"


def test_a_mind_resume_is_kept_as_ouroboros_and_its_own_outcome_event_stays(tmp_path, monkeypatch):
    from supervisor.events_mind_controls import handle_mind_task_control

    _queue, _workers, successor = _paused_held_continue(tmp_path, monkeypatch)
    pushed = []
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, bridge=SimpleNamespace(push_log=pushed.append))
    handle_mind_task_control({"task_id": successor, "action": "resume_task", "actor_task_id": "coordinator-1",
                              "action_nonce": "resume-0001", "reason": "Restore the saved work."}, ctx)
    [outcome] = _events(tmp_path, "mind_task_control_outcome")
    assert outcome["outcome"]["error"] == "predecessor_writers_unsettled" and pushed == [outcome]
    assert _events(tmp_path, "task_resume_outcome") == [], "the mind door keeps its own original"
    stored = _stored(tmp_path, successor)
    assert stored["by"] == {"actor": "ouroboros", "task_id": "coordinator-1"}
    assert stored["answer"]["message"] == outcome["outcome"]["message"]
    work = _work(tmp_path, successor)
    assert work["last_resume"]["by"] == {"actor": "ouroboros"} and work["last_resume"]["text"] == FACTS
    assert _without_attempt(work) == _work(tmp_path, successor, attempts=False), "no turn, control or word of its own"


def test_a_childs_refused_selection_names_who_selected_it_and_the_owners_selection_ends_it(tmp_path, monkeypatch):
    """A member held under its root's money latch: the root's model-issued selection is refused for want of the root's
    own Resume grant (Q9). The member hold that refusal wrote sits under the same money latch, the control the
    selection answered about, so it is said; the owner's own selection releases the member and keeps its answer."""
    from ouroboros.gateway.tasks import api_task_resume
    from ouroboros.task_results import STATUS_SCHEDULED, write_task_result
    from supervisor.events_budget import _handle_budget_resume_child, _set_root_budget_pause_locked
    from tests._budget_pause_exact_helpers import _install_queue

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    _set_root_budget_pause_locked("root-f2", {"scope": "root", "root_task_id": "root-f2"})
    member = {"id": "m1-f2", "type": "task", "chat_id": 0, "root_task_id": "root-f2", "parent_task_id": "root-f2",
              "_attempt": 1, "admitted_dispatch": "none"}
    workers.PENDING.append(member)
    write_task_result(tmp_path, "m1-f2", STATUS_SCHEDULED, root_task_id="root-f2", parent_task_id="root-f2")
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, PENDING=workers.PENDING, RUNNING=workers.RUNNING,
                          bridge=SimpleNamespace(push_log=lambda _event: None))
    _handle_budget_resume_child({"task_id": "m1-f2", "requested_by": "root-f2", "reason": "continue its part"}, ctx)
    [event] = _events(tmp_path, "budget_resume_child_outcome")
    assert event["error"] == "root_resume_grant_missing", event
    stored = _stored(tmp_path, "m1-f2")
    assert (stored["answer"]["error"], stored["by"], stored["selected_by"]) == (
        "root_resume_grant_missing", {"actor": "ouroboros", "task_id": "root-f2"}, "root-f2")
    assert stored["control"]["cause"] == "money" and stored["control"]["scope"] == "tree"
    [work] = [work for work in state_fields(tmp_path)["works"] if "m1-f2" in work["members"]["ids"]]
    held = work["members"]["states"]["m1-f2"]
    assert held["activity"] == "paused" and held["last_resume"]["error"] == "root_resume_grant_missing"
    assert held["last_resume"]["by"] == {"actor": "ouroboros"}
    status, selected = _http(api_task_resume, "m1-f2")
    assert status == 200 and selected["ok"] is True, selected
    assert _stored(tmp_path, "m1-f2")["ok"] is True and _stored(tmp_path, "m1-f2")["by"] == {"actor": "owner"}


def _held_row(task_id, pause_cause, **extra):
    return {"activity_id": task_id, "phase": "budget_paused", "kind": "managed_task", "chat_id": 42, "task_attempt": 1,
            "started_at": 100.0, "pause_cause": pause_cause, **extra}


SAVED = {"_budget_pause_hold": {"reason": "saved_work_hold", "release": "resume", "held_at": "2026-10-17T14:00:00Z",
                                "by": {"actor": "host", "task_id": "", "text": "", "at": "2026-10-17T14:00:00Z"}}}
# The control the saved hold is (``work_stop.control_identity``), as a door asked about it would have read it.
SAVED_CONTROL = {"cause": "app_stop", "scope": "member", "ui_code": "saved_work_hold", "reason": "saved_work_hold",
                 "release": "resume", "held_at": "2026-10-17T14:00:00Z"}


def _refused(control=SAVED_CONTROL, at="2026-10-17T15:05:00Z", **extra):
    return {"at": at, "answered_at": at, "by": {"actor": "owner"}, "control": control, "ok": False,
            "answer": {"ok": False, "error": "predecessor_writers_unsettled",
                       "message": "Resume refused: task h1 is queued to run"}, **extra}


@pytest.mark.parametrize("attempt, said", [
    # About this very hold, whenever it was answered: dated history beside it.
    (_refused(), True), (_refused({"cause": "pause", "scope": "tree", "ui_code": "owner_paused", "release": "resume",
               "fence_id": "fence-earlier"}), False),
    (_refused({**SAVED_CONTROL, "reason": "owner_restart_hold", "ui_code": "owner_restart_hold"}), False),
    # An earlier stop in the very same words (a stop declared anew, after a release, records its own occurrence).
    (_refused({**SAVED_CONTROL, "held_at": "2026-10-16T09:00:00Z"}), False),
    # No control named, accepted, malformed, an unknown asker, unordered: nothing to say.
    (_refused({}), False), ({**_refused(), "ok": True}, False), ({"ok": False}, False), ("refused", False),
    ({**_refused(), "by": {"actor": "stranger"}}, False), ({**_refused(), "at": "not a time"}, False),
])
def test_an_attempt_is_said_only_beside_the_control_it_answered_about(attempt, said):
    fields = work_fields([_held_row("sr", "restart", can_resume=True)], {"sr": SAVED}, attempts={"sr": attempt})
    [work] = fields["works"]
    assert work["because"][0]["code"] == "saved_on_stop" and {"action": "resume", "target": "sr"} in work["controls"]
    assert ("last_resume" in work) is said, work
    plain = work_fields([_held_row("sr", "restart", can_resume=True)], {"sr": SAVED})["works"][0]
    assert _without_attempt(work) == plain, "the attempt changes nothing else the work says or offers"


def test_only_held_members_read_an_attempt_and_an_unreadable_one_is_a_disclosed_gap(tmp_path, monkeypatch):
    """The census's body facts carry the attempt; the projection asks for it only for a held member. A store that
    cannot answer is a gap the projection names, never an absent attempt: the work keeps its state and its doors."""
    from ouroboros.gateway import state as gateway_state
    from ouroboros.gateway import work_projection as wp

    asked = []
    monkeypatch.setattr(gateway_state, "_task_activity_facts",
                        lambda root, task_id: asked.append(task_id) or {"unavailable": True})
    rows = [_held_row("sr", "restart", can_resume=True),
            {"activity_id": "q1", "phase": "queued", "kind": "managed_task", "chat_id": 42, "task_attempt": 1,
             "started_at": 50.0}]
    import contextlib
    import pathlib
    import tempfile

    from ouroboros import store

    with tempfile.TemporaryDirectory() as root, contextlib.ExitStack() as release, \
            mock.patch.object(wp._Observations, "result", lambda self, task_id: None), \
            mock.patch.object(wp._Observations, "fence", lambda self, root_id, drive_root=None: {}), \
            mock.patch.object(wp._Observations, "model_handoffs", lambda self, root_id, drive_root=None: []), \
            mock.patch.object(wp, "_open_runs", lambda drive_root: {}), \
            mock.patch.object(wp, "_inflight_upkeep", lambda drive_root, known: []):
        # As ``work_fields``: the projection's real reads cache this root's stores (its usage store among them) and
        # Windows removes no open file, so this root's handles go back before the root does.
        release.callback(store.close_connections_under, pathlib.Path(root))
        snapshot = json.loads(json.dumps(wp.works_snapshot(root, rows, {"sr": SAVED})))
    assert asked == ["sr"], "only the held member's attempt is read"
    assert snapshot["works_gaps"] == [{"reason": "resume_attempt_unavailable", "task_id": "sr"}]
    held = next(work for work in snapshot["works"] if work["episode"]["task_id"] == "sr")
    assert held["confirmed"] is True and "last_resume" not in held
    assert {"action": "resume", "target": "sr"} in held["controls"], "history unread is no change of state"


def test_the_census_reads_an_attempt_once_per_row_change_and_writes_nothing(tmp_path, monkeypatch):
    """Passive state: the attempt rides the census's memoized body facts (``_task_activity_facts``), so a second read
    of an unchanged row decodes no body, and neither read writes; a new attempt moves the row and is read once."""
    from ouroboros import task_store
    from ouroboros.gateway.tasks import api_task_resume

    _queue, _workers, successor = _paused_held_continue(tmp_path, monkeypatch)
    status, _refusal = _http(api_task_resume, successor)
    assert status == 409
    real, decoded = task_store.read_document, []

    def counting(root, task_id, *args, **kwargs):
        decoded.append(task_id)
        return real(root, task_id, *args, **kwargs)

    monkeypatch.setattr(task_store, "read_document", counting)
    version = task_store.task_versions(tmp_path, [successor])
    first = _work(tmp_path, successor)
    reads = decoded.count(successor)
    second = _work(tmp_path, successor)
    assert first["last_resume"] == second["last_resume"] and decoded.count(successor) == reads, "no second decode"
    assert task_store.task_versions(tmp_path, [successor]) == version, "the census wrote nothing"
    status, _again = _http(api_task_resume, successor)
    assert status == 409 and task_store.task_versions(tmp_path, [successor]) != version
    third = _work(tmp_path, successor)
    assert decoded.count(successor) == reads + 1 and third["last_resume"]["at"] > first["last_resume"]["at"]


def test_a_kept_attempt_is_the_doors_whole_answer_and_keeps_no_acceptance(tmp_path, monkeypatch):
    """A refusal whose every holder is an unknown held call offers ``acceptable_unknown`` in the door's answer alone:
    the kept attempt holds every blocker and every word the door gave (nothing cut), never ids to go on past later."""
    from ouroboros.control_actor import owner_door
    from ouroboros.task_results import STATUS_SCHEDULED, write_task_result
    from ouroboros.work_stop import resume_asked
    from tests._budget_pause_exact_helpers import _install_queue

    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "unknown-1", STATUS_SCHEDULED)
    blockers = [{"kind": "tool_handoff", "task_id": "unknown-1", "tool": "shell", "reason": "owner_ended_effect_unknown",
                 "operation_id": f"op-{n}", "effects": ["files"], "detail": "x" * 5000} for n in range(12)]
    answer = {"ok": False, "error": "owner_pause_effects_unsettled", "blockers": blockers,
              "acceptable_unknown": [f"op-{n}" for n in range(12)], "message": "Resume refused: " + "y" * 2000}
    with owner_door("api_task_resume"):
        answered = resume_asked(queue, "unknown-1")
    assert answered(answer) is answer
    stored = _stored(tmp_path, "unknown-1")
    assert "acceptable_unknown" not in stored["answer"] and stored["by"] == {"actor": "owner"}
    assert stored["answer"] == {key: value for key, value in answer.items() if key != "acceptable_unknown"}
    [event] = _events(tmp_path, "task_resume_outcome")
    assert event["outcome"] == answer, "the event history keeps the whole answer as the door gave it"
