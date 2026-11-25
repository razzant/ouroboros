"""A never-started Continue may name its predecessor's exact unbound read-only start (owner, 24 Nov).

Real custody rows (``delegate_custody``), the real Continue admission, the real owner Resume door over HTTP (both
of its held-Continue seams: the writer hold and the owner Pause's fence), the real census and the real recovered
start binding run here; no engine, model or network is involved and every identity is fictional. The episode ends
as a real owner Restart records it (``_stop_record``). The choice is offered only when the whole predecessor
census is such starts and custody proves each inherited grant read-only, back to the first run. Only a Resume
addressed alone (the owner's, or one ``resume_task`` naming it) can take the choice, never a list or Panic row.
It is recorded for that successor and its current stop only; retries of the old request only look it up
(its handshake never becomes an admitting receipt), and the start becomes the successor's own custody. Its own
Continue inherits no decision: only that Continue's own addressed Resume, through the verified chain, names the start
again. The first proof of the old run's activity holds each successor that named it again under its launch lock (an
answered one while its tree still runs), never waits for that lock, and leaves the proof owed when it cannot read them.
"""
from __future__ import annotations

import contextlib
import json
from urllib.parse import quote

import pytest

from ouroboros import delegate_custody as dc
from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx
from tests._owner_continue_helpers import NONCE, _running_predecessor
from tests._owner_pause_ended_helpers import _resume, _tick

PRED = "pred-1"  # the interrupted episode, ended by a real owner Restart (``_stop_record``)
LOST = "inv-lost"  # its helper's continuation request whose reply was never confirmed
HEAD = "run-head"  # the finished read-only run that request continues
STORE = "store-fictional"
BODY = {"prompt": "Keep reading the code.", "instructions": "Read only; report findings.",
        "authPreference": "subscription", "mode": "ask", "scope": {"kind": "project", "root": "/fictional/checkout"},
        "harnesses": ["codex"], "primaryHarness": "codex", "access": "readonly"}


def _stop_record(task_id, source="owner_restart", **extra):
    """What a settled cancel records on the work a stop ended, from its own producers: the cancel intent the stop mints
    (a real owner Restart's: ``server_restart`` calls ``request_cancel(source="owner_restart", requested_by="owner")``,
    its id included) as ``cancel_publication._intent_outcome_fields`` keeps it."""
    from supervisor.cancel_publication import _intent_outcome_fields

    return _intent_outcome_fields({"request_id": "ci_0a1b2c3d4e5f", "task_id": task_id, "state": "requested",
                                   "reason": "Owner restart", "source": source, "requested_by": "owner",
                                   "requested_at": "2026-11-20T09:00:00+00:00", "generation": 0, "scope": "single",
                                   **extra})


def _head(root, *, access="readonly", tree=PRED, store=STORE, execution=None):
    """The run the lost request continues: requested and bound with ``access`` (and its own ``execution`` group, when
    it sent one) in ``tree``'s custody, then settled."""
    if tree != PRED:
        _running_predecessor(root, tree)  # another task's own tree, with its own billing authority
    assert dc.record_start_requested(
        root, task_id=tree, root_task_id=tree, invocation_id="inv-head", idempotency_key="inv-head",
        request={**BODY, "access": access, **({"execution": execution} if execution else {})}, route="codex",
        project_id="", project_owned=False,
        daemon_store_id=store, store_identity_unproven=False, effective_access=access, execution_mode="ask")
    assert dc.record_started(root, dc.RunCustody(
        run_id=HEAD, task_id=tree, root_task_id=tree, invocation_id="inv-head", route_id="codex", model="m",
        ledger_root=str(root), daemon_store_id=store), shape={"access": access, "mode": "ask", "isolation": "",
                                                             "delegated": False})
    assert dc.emit(root, dc.SETTLED, {"run_id": HEAD, "task_id": tree, "route": "codex", "model": "m",
                                      "state": "succeeded"})


def _lost(root, *, starter=PRED, body=None, drop=(), **facts):
    """The pending continuation request as the incident recorded it: hash-addressed body plus the host's own facts,
    with no admitting-store receipt (``daemon_store_id`` empty, unproven), only the store its handshake saw."""
    request = {**BODY, "continueFrom": HEAD, **(body or {})}
    for key in drop:
        request.pop(key, None)
    fields = {"daemon_store_id": "", "handshake_store_id": STORE, "store_identity_unproven": True,
              "continuation_of": HEAD, "effective_access": "readonly", "execution_mode": "ask", **facts}
    assert dc.record_start_requested(
        root, task_id=starter, root_task_id=starter, invocation_id=LOST, idempotency_key=LOST, request=request,
        route="codex", project_id="", project_owned=False, **{k: v for k, v in fields.items() if v is not None})


def _held_continue(root, monkeypatch, *, head=None, lost=None, before=None):
    """The incident's shape: the episode made one unconfirmed read-only start (``before``: what else it did meanwhile),
    a real owner Restart cut it (``_stop_record``), and its Continue waits for that start."""
    from ouroboros.task_results import STATUS_CANCELLED, write_task_result
    from supervisor import budget_resume
    from supervisor.continuation_admission import admit_continuation

    queue, state, workers = _install_queue(root, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    # The work's money authority has its own case below (``test_the_choice_meets_the_grants_money_refusal``).
    monkeypatch.setattr(budget_resume, "_root_money_refusal", lambda *_a, **_k: None)
    _running_predecessor(root, PRED)
    _head(root, **(head or {}))
    _lost(root, **(lost or {}))
    if before is not None:
        before(root)
    write_task_result(root, PRED, STATUS_CANCELLED, result="interrupted", **_stop_record(PRED))
    dc._CUSTODY.clear()
    ack = admit_continuation(PRED, action_nonce=NONCE)
    assert ack["ok"] and ack["held"], ack
    return queue, workers, ack["successor_task_id"]


def _decisions(root):
    return dc.invocation_record(root, LOST).get("resume_decisions") or {}


def _rows(root, kind):
    return [row for row in dc.custody_rows(root) if row["type"] == kind and row.get("invocation_id") == LOST]


def _own_earlier_choice(root):
    """The episode's own earlier lookup-only decision about the same start (resuming the episode itself), taken before
    the Restart cut it: no choice is recorded over a root a recorded stop ended (``still_pending``)."""
    from ouroboros.delegate_resume import _record, address

    start = address({"task_id": PRED, "root_task_id": PRED, "invocation_id": LOST})
    assert _record(root, start, PRED, {"actor": "owner"}, "earlier choice", choice_id=start["choice_id"] + f"/{PRED}//")


def test_the_successor_names_the_proven_read_only_start_once_and_carries_it(tmp_path, monkeypatch):
    from ouroboros.delegate_resume import LOOKUP_ONLY, StartHeld, _admit_start
    from ouroboros.task_results import load_task_result
    from ouroboros.work_stop import hold_fact
    from supervisor.continuation_admission import CONTINUE_PURPOSE, conflicting_writers

    queue, workers, successor = _held_continue(tmp_path, monkeypatch, before=_own_earlier_choice)
    refusal = _resume("owner", selected=successor)
    assert refusal["error"] == "predecessor_writers_unsettled" and "action" not in refusal, refusal
    choice, = refusal["acceptable_unknown"]
    assert choice.startswith(f"delegate_start:{PRED}/{PRED}/{LOST}/{quote(successor, safe='')}///hold"), choice
    blocker, = refusal["blockers"]
    assert blocker["read_only"] is True and blocker["choice_id"] == choice
    assert set(_decisions(tmp_path)) == {PRED}, "the episode's own earlier decision grants its successor nothing"

    ok = _resume("owner", [choice], selected=successor)
    assert ok["ok"] and ok["selection"] == "budget_hold_released", ok
    risk, = ok["accepted_start_risks"]
    assert (risk["successor_of"], risk["resuming_task_id"], risk["creation"]) == (PRED, successor, "lookup_only")
    row = next(item for item in workers.PENDING if item["id"] == successor)
    assert hold_fact(row) is None, "released from its writer hold"
    assert set(_decisions(tmp_path)) == {PRED, successor} and len(_rows(tmp_path, dc.START_RISK_ACCEPTED)) == 2
    stored = load_task_result(tmp_path, successor)
    assert stored["accepted_predecessor_starts"] == {LOST: {"root_task_id": PRED, "task_id": PRED, "invocation_id": LOST}}
    notice, = [note for key, note in stored["pause_notices"].items() if key.endswith(choice)]
    from ouroboros.pause_notices import notice_text
    assert "earlier request was read-only" in notice_text(notice) and "changing files" not in notice_text(notice)
    with pytest.raises(StartHeld) as held:
        with _admit_start(tmp_path, PRED, LOST):
            pytest.fail("the old request is never sent again")
    assert held.value.args[0] == LOOKUP_ONLY
    from ouroboros.delegate_pending import replay_store_unproven
    record = dc.invocation_record(tmp_path, LOST)
    assert (record["state"], record["daemon_store_id"], record["handshake_store_id"], record["store_identity_unproven"]) \
        == ("pending", "", STORE, True), "the handshake stays a handshake: no admitting receipt was minted"
    assert replay_store_unproven(record, STORE), "an unproven admission is still never replayed"
    assert not any(key in risk for key in ("daemon_store_id", "handshake_store_id")), risk
    # While the old start stays unknown it is this successor's own accepted custody; its own Continue inherits nothing.
    own = conflicting_writers(queue, successor, drive_root=tmp_path)
    assert not any(b.get("invocation_id") == LOST for b in own), own
    later = conflicting_writers(queue, successor, drive_root=tmp_path, purpose=CONTINUE_PURPOSE)
    assert any(b.get("invocation_id") == LOST and b["stop_outcome"] == "pending_invocation_unbound" for b in later)


def test_the_waiting_continue_offers_resume_as_a_check_and_only_the_confirmed_press_records_the_choice(
        tmp_path, monkeypatch):
    """The real census and work projection of the held Continue offer its addressed Resume, because the wait records
    only this unconfirmed start; the real door then answers that press with the choice. The choice records nothing
    until it comes back with a reason, so a press the owner cancels leaves no decision and the next one asks again."""
    from ouroboros.work_stop import hold_fact
    from tests.test_resume_attempt_history import _work

    _queue, workers, successor = _held_continue(tmp_path, monkeypatch)
    work = _work(tmp_path, successor)
    assert (work["activity"], work["because"][0]["code"]) == ("waiting", "earlier_work_unconfirmed")
    assert {"action": "resume", "target": successor} in work["controls"]
    assert not any(control["action"] == "pause" for control in work["controls"]), "a wait is never Paused from here"

    asked = _resume("owner", selected=successor)  # the press: the door counts afresh and offers the choice
    choice, = asked["acceptable_unknown"]
    assert asked["blockers"][0]["read_only"] is True and asked["blockers"][0]["choice_id"] == choice
    assert not _decisions(tmp_path) and not _rows(tmp_path, dc.START_RISK_ACCEPTED), "asking records nothing"
    assert _resume("owner", selected=successor)["acceptable_unknown"] == [choice], "a cancelled press asks again"
    assert not _decisions(tmp_path)

    ok = _resume("owner", [choice], selected=successor)  # Resume anyway: the owner's reason travels with the id
    assert ok["ok"] and ok["selection"] == "budget_hold_released", ok
    assert set(_decisions(tmp_path)) == {successor}
    assert hold_fact(next(item for item in workers.PENDING if item["id"] == successor)) is None


@pytest.mark.parametrize("case", [
    "mode_absent", "agent_mode", "writable_execution", "thread_turn", "project_input", "continue_unrecorded",
    "host_access_missing", "store_unproven", "head_writer", "head_other_tree", "head_other_store", "running_leaf",
])
def test_without_positive_no_write_proof_the_predecessor_refusal_stays(tmp_path, monkeypatch, case):
    lost = {"mode_absent": {"drop": ("mode",)},
            "agent_mode": {"body": {"mode": "agent"}},
            "writable_execution": {"body": {"execution": {"isolation": "live", "delegated": True,
                                                          "workspaceRoot": "/fictional/checkout"}}},
            "thread_turn": {"body": {"_thread_id": "thread-fictional"}},
            "project_input": {"body": {"projectId": "project-fictional"}},
            "continue_unrecorded": {"continuation_of": None},
            "host_access_missing": {"effective_access": None},
            "store_unproven": {"daemon_store_id": "", "handshake_store_id": ""}}.get(case)  # no store fact at all
    head = {"head_writer": {"access": "workspace_write"}, "head_other_tree": {"tree": "other-root"},
            "head_other_store": {"store": "store-other"}}.get(case)

    def leaf(root):
        assert dc.record_started(root, dc.RunCustody(run_id="run-leaf", task_id=PRED, root_task_id=PRED,
                                                     route_id="codex", model="m", ledger_root=str(root)))

    _queue, _workers, successor = _held_continue(tmp_path, monkeypatch, head=head, lost=lost,
                                                 before=leaf if case == "running_leaf" else None)
    refusal = _resume("owner", selected=successor)
    assert refusal["error"] == "predecessor_writers_unsettled", refusal
    assert "acceptable_unknown" not in refusal and not any(b.get("choice_id") for b in refusal["blockers"])
    assert not _decisions(tmp_path)


def test_the_choice_meets_the_grants_money_refusal(tmp_path, monkeypatch):
    from supervisor import budget_resume

    _queue, _workers, successor = _held_continue(tmp_path, monkeypatch)
    monkeypatch.setattr(budget_resume, "_root_money_refusal",
                        lambda *_a, **_k: {"ok": False, "error": "root_hard_cap_exhausted",
                                           "action": "increase_budget_then_resume"})
    refusal = _resume("owner", selected=successor)
    assert refusal["error"] == "root_hard_cap_exhausted" and "acceptable_unknown" not in refusal, refusal
    assert not _decisions(tmp_path)


def test_the_owner_paused_successor_binds_its_choice_to_that_pause_and_a_new_pause_asks_again(tmp_path, monkeypatch):
    from ouroboros import owner_pause
    from supervisor.owner_pause_control import request_owner_pause

    queue, _workers, successor = _held_continue(tmp_path, monkeypatch)
    by = {"actor": "owner", "text": "", "at": "2026-11-25T00:00:00+00:00"}
    assert request_owner_pause(successor, request_id="pause-1", by=by)["ok"]
    fence = owner_pause.read_fence(tmp_path, successor)
    first, = _resume("owner", selected=successor)["acceptable_unknown"]
    assert first.startswith(f"delegate_start:{PRED}/{PRED}/{LOST}/{quote(successor, safe='')}/{fence['fence_id']}/")
    ok = _resume("owner", [first], selected=successor)
    assert ok["ok"] and ok["owner_pause_released"] and ok["never_started"], ok
    assert not owner_pause.fence_closed(owner_pause.read_fence(tmp_path, successor))

    assert request_owner_pause(successor, request_id="pause-2", by=by)["ok"]
    _tick(queue)
    second, = _resume("owner", selected=successor)["acceptable_unknown"]
    assert second != first, "a newer Pause needs its own decision on the same unknown start"
    assert len(_rows(tmp_path, dc.START_RISK_ACCEPTED)) == 1


def test_a_failed_release_reuses_its_recorded_choice_for_the_same_hold(tmp_path, monkeypatch):
    queue, _workers, successor = _held_continue(tmp_path, monkeypatch)
    choice, = _resume("owner", selected=successor)["acceptable_unknown"]
    publish = queue.publish_queue_snapshot
    failures = iter([True])

    def flaky(*args, **kwargs):
        return False if next(failures, False) else publish(*args, **kwargs)

    with monkeypatch.context() as torn:
        torn.setattr(queue, "publish_queue_snapshot", flaky)
        refused = _resume("owner", [choice], selected=successor)
    assert refused.get("error") == "snapshot_not_persisted", refused
    assert len(_rows(tmp_path, dc.START_RISK_ACCEPTED)) == 1, "the choice was recorded before the release failed"
    ok = _resume("owner", [choice], selected=successor)
    assert ok["ok"], ok
    assert len(_rows(tmp_path, dc.START_RISK_ACCEPTED)) == 1, "the same hold's retry records nothing twice"


def _list_row(root, successor, ids=()):
    """One row of the mind's list Resume (``resume_task(task_ids=[...])``): the call's named ids ride every row."""
    from ouroboros.task_results import write_task_result
    from supervisor.events_mind_controls import _apply

    write_task_result(root, "mind-root", "running", root_task_id="mind-root")  # the list call's live sender
    return _apply({"task_id": successor, "action": "resume_task", "actor_task_id": "mind-root",
                   "action_nonce": "bulk-row", "reason": "resume the listed works", "accept_unknown": list(ids),
                   "list_call": "call-bulk", "general": True})


@pytest.mark.parametrize("bulk", ["shown_key", "saved_choice"])
def test_a_list_or_panic_resume_never_offers_records_or_reuses_the_successors_choice(tmp_path, monkeypatch, bulk):
    from ouroboros.control_actor import accepting_unknown, owner_door
    from ouroboros.delegate_resume import predecessor_starts, with_resume_decisions
    from ouroboros.work_stop import hold_fact
    from supervisor.budget_resume_unstarted import accept_predecessor_starts
    from supervisor.continuation_admission import CONTINUE_PURPOSE, conflicting_writers

    queue, workers, successor = _held_continue(tmp_path, monkeypatch)
    choice, = _resume("owner", selected=successor)["acceptable_unknown"]
    if bulk == "saved_choice":  # an addressed press recorded it, then its release failed: the choice stays saved
        publish = queue.publish_queue_snapshot
        failures = iter([True])
        with monkeypatch.context() as torn:
            torn.setattr(queue, "publish_queue_snapshot",
                         lambda *a, **k: False if next(failures, False) else publish(*a, **k))
            assert _resume("owner", [choice], selected=successor).get("error") == "snapshot_not_persisted"
    saved = len(_rows(tmp_path, dc.START_RISK_ACCEPTED))
    assert saved == (1 if bulk == "saved_choice" else 0)

    keys = [choice] if bulk == "shown_key" else []
    row = next(item for item in workers.PENDING if item["id"] == successor)
    writers = conflicting_writers(queue, PRED, drive_root=tmp_path, purpose=CONTINUE_PURPOSE)
    proof = predecessor_starts(tmp_path, PRED, successor, writers)
    assert proof == {LOST: PRED}, "an individual Resume could consider this exact evidence"
    listed = _list_row(tmp_path, successor, keys)
    # The Panic sheet's HTTP shape and unblocked path have their own gateway tests. Exercise
    # the shared consumer with its group marker even if it already has the predecessor facts.
    with owner_door("api_task_resume"), accepting_unknown(keys, "Choose one saved work"):
        panic = with_resume_decisions(successor, lambda: accept_predecessor_starts(
            row, {"predecessor_starts": proof}, writers, tmp_path, fence_id=""), panic_selection=True)
    for refused in (listed, panic):
        assert refused.get("ok") is False and refused["error"] == "predecessor_writers_unsettled", refused
        assert not refused.get("acceptable_unknown") and not any(b.get("choice_id") for b in refused.get("blockers", ()))
    assert len(_rows(tmp_path, dc.START_RISK_ACCEPTED)) == saved, "a group row records no choice"
    assert hold_fact(row) is not None, "no group row reuses a saved choice to release it"

    ok = _resume("mind", [choice], selected=successor)  # one ordinary resume_task(task_id=...) with the key and a reason
    assert ok["ok"], ok
    assert len(_rows(tmp_path, dc.START_RISK_ACCEPTED)) == 1


def _bind_late(root, state="running"):
    """A later lookup finds the old request accepted after all: one exact run, bound with its first state."""
    from ouroboros.delegate_accepted import bind_started

    entry = dc.RunCustody(run_id="run-late", task_id=PRED, root_task_id=PRED, invocation_id=LOST, route_id="codex",
                          model="m", ledger_root=str(root), daemon_store_id=STORE)
    return bind_started(root, entry, observed_state=state)


@pytest.mark.parametrize("own_choice,first_state", [(False, "running"), (True, "running"), (False, "")],
                         ids=["successor_only", "with_the_episodes_own_choice", "bound_unknown_then_observed"])
def test_late_activity_holds_the_successor_under_its_launch_lock_until_the_old_run_settles(
        tmp_path, monkeypatch, own_choice, first_state):
    from ouroboros import delegate_resume, owner_pause
    from ouroboros.delegate_custody_current import run
    from ouroboros.store_hot_facts import recovered_owner_pause_roots
    from supervisor.continuation_admission import CONTINUE_PURPOSE, conflicting_writers

    queue, _workers, successor = _held_continue(tmp_path, monkeypatch, before=_own_earlier_choice if own_choice else None)
    choice, = _resume("owner", selected=successor)["acceptable_unknown"]
    assert _resume("owner", [choice], selected=successor)["ok"]

    order, real_lock = [], owner_pause.launch_lock

    @contextlib.contextmanager
    def recorded(root, root_task_id, **kwargs):
        order.append(str(root_task_id))
        with real_lock(root, root_task_id, **kwargs):
            yield

    with monkeypatch.context() as locks:
        locks.setattr(owner_pause, "launch_lock", recorded)
        assert _bind_late(tmp_path, first_state).durable
        if not first_state:  # bound with no state yet: no hold and no lock until its first actual status
            assert not order and not owner_pause.fence_closed(owner_pause.read_fence(tmp_path, successor))
            assert delegate_resume.observe_bound_state(tmp_path, run(tmp_path, "run-late"), "running") is True
            assert len(_rows(tmp_path, dc.START_STATE_OBSERVED)) == 1
    assert order[:2] == [PRED, successor], "the tree's launch lock, then the successor's, before the write"

    fence = owner_pause.read_fence(tmp_path, successor)
    start = fence["active_delegated_start"]
    assert owner_pause.fence_closed(fence) and fence["observe_only"], fence
    assert (start["holder_task_id"], start["root_task_id"], start["invocation_id"], start["run_id"]) == (
        successor, PRED, LOST, "run-late")
    assert not owner_pause.fence_closed(owner_pause.read_fence(tmp_path, PRED)), \
        "the Restart's own stop owns what the ended tree left: no choice mints a control on it"
    assert successor in recovered_owner_pause_roots(tmp_path), "boot rediscovers the successor's fence"
    ctx, _limit = _loop_ctx(tmp_path, successor)
    with pytest.raises(owner_pause.OwnerPauseRefused):
        owner_pause.run_operation(ctx, lambda: pytest.fail("the successor launched past the old run's activity"))
    for purpose in ("", CONTINUE_PURPOSE):  # its own Resume and its own Continue both wait for the exact old run
        census = conflicting_writers(queue, successor, drive_root=tmp_path, purpose=purpose)
        assert any(b.get("run_id") == "run-late" and b["state"] == "running" for b in census), census

    _tick(queue)
    waiting = _resume("owner", selected=successor)
    assert waiting["ok"] and waiting.get("held"), waiting
    assert any(b.get("run_id") == "run-late" for b in waiting["blockers"])

    woken = []
    monkeypatch.setattr(delegate_resume, "recovered_writer_changed", lambda _root, tree: woken.append(tree) or True)
    assert dc.emit(tmp_path, dc.SETTLED, {"run_id": "run-late", "task_id": PRED, "route": "codex", "model": "m",
                                          "state": "completed"})
    assert successor in woken and PRED in woken, woken
    dc._CUSTODY.clear()
    assert not any(b.get("run_id") == "run-late"
                   for b in conflicting_writers(queue, successor, drive_root=tmp_path, purpose=CONTINUE_PURPOSE))


def test_activity_without_the_successors_launch_lock_lands_nothing(tmp_path, monkeypatch):
    from ouroboros import delegate_resume, owner_pause

    _queue, _workers, successor = _held_continue(tmp_path, monkeypatch)
    choice, = _resume("owner", selected=successor)["acceptable_unknown"]
    assert _resume("owner", [choice], selected=successor)["ok"]

    @contextlib.contextmanager
    def tree_only(root, event):
        with owner_pause.launch_lock(root, event.get("root_task_id") or event["task_id"]):
            yield ()

    monkeypatch.setattr(delegate_resume, "publication_lock", tree_only)
    assert not _bind_late(tmp_path).durable, "no activity row lands outside the successor's launch lock"
    assert dc.invocation_record(tmp_path, LOST)["state"] == "pending"
    assert not owner_pause.fence_closed(owner_pause.read_fence(tmp_path, successor))


def test_boot_discovery_reads_a_successor_fence_by_its_holder():
    from ouroboros.store_hot_facts import _recovered_start_root

    fence = {"state": "requested", "active_delegated_start": {"root_task_id": "pred-x", "holder_task_id": "succ-x"}}
    row = {"control_valid": 1, "owner_pause_state": "requested", "owner_pause_facts": json.dumps(fence)}
    assert _recovered_start_root({**row, "task_id": "succ-x"}) == "succ-x"
    assert _recovered_start_root({**row, "task_id": "pred-x"}) == "pred-x"
    assert _recovered_start_root({**row, "task_id": "other-x"}) == ""


@pytest.mark.parametrize("source,extra,ended", [
    ("owner_restart", {}, False), ("snapshot_restore", {"reason": "server_shutdown"}, False),
    ("server_shutdown", {"reason": "server_shutdown"}, False), ("http_single", {"reason": "Stopped"}, True),
    ("panic", {"reason": "Panic"}, True), ("unrecorded_cause", {"reason": "Stopped"}, True)])
def test_a_stop_is_read_by_its_recorded_class_never_by_its_intents_id(source, extra, ended):
    """Every one of these records carries its intent's id; only my human's Stop or Panic (or a cause Continue would
    not recognize) ends a chain's root for a later choice, never the technical interruption a Continue resumes."""
    from ouroboros.delegate_resume import _owner_ended
    from ouroboros.owner_pause import owner_stopped

    row = {"status": "cancelled", **_stop_record(PRED, source, **extra)}
    assert owner_stopped(row), "the bare id check alone would take each of them for a Stop"
    assert _owner_ended(row) is ended


@pytest.mark.parametrize("ended_by,recorded", [("owner_restart", True), ("http_single", False), ("panic", False)])
def test_the_write_takes_a_successor_choice_only_over_its_own_lineage_and_never_past_a_stop(
        tmp_path, monkeypatch, ended_by, recorded):
    """The one authoritative check of the addressed POST, in the choice's own write (``still_pending``): the chain from
    the start's tree to the choosing Continue verified hop by hop, its earlier roots read by the class of their stop."""
    from ouroboros import store
    from ouroboros.delegate_resume import predecessor_starts, still_pending
    from ouroboros.task_results import STATUS_CANCELLED, write_task_result
    from supervisor.continuation_admission import CONTINUE_PURPOSE, conflicting_writers

    queue, _workers, successor = _held_continue(tmp_path, monkeypatch)
    blockers = conflicting_writers(queue, PRED, drive_root=tmp_path, purpose=CONTINUE_PURPOSE)
    assert predecessor_starts(tmp_path, PRED, successor, blockers) == {LOST: PRED}
    assert predecessor_starts(tmp_path, PRED, "rogue-root", blockers) == {}, "no verified Continue reaches it"
    assert predecessor_starts(tmp_path, "rogue-root", successor, blockers) == {}, "nor a start it never carried"
    if ended_by != "owner_restart":  # synthetic: the same ended row, as my human's own stop would have recorded it
        write_task_result(tmp_path, PRED, STATUS_CANCELLED, result="interrupted",
                          **_stop_record(PRED, ended_by, reason="Stopped", request_id="ci_5f4e3d2c1b0a"))

    def event(selected):
        return {"root_task_id": PRED, "task_id": PRED, "invocation_id": LOST, "resuming_task_id": selected,
                "decision": {"successor_of": PRED, "fence_id": "", "pause_id": ""}}

    with store.open_store(tmp_path, write=False) as txn:
        assert still_pending(txn, event(successor)) is recorded
        assert not still_pending(txn, event("rogue-root"))


@pytest.mark.parametrize("case", [
    "omitted", "empty", "explicit", "explicit-no-aux", "old-writer", "prior-body-unavailable",
])
def test_old_start_proof_does_not_infer_a_whole_group_override(tmp_path, monkeypatch, case):
    """The saved transport's build is unproven: the read-only chain must stand even if it inherited execution."""
    from ouroboros import delegate_pending

    group = {"isolation": "envelope", "delegated": False, "auxiliaryWritableRoots": []}
    execution = {key: value for key, value in group.items() if key != "auxiliaryWritableRoots"}
    sent = {"empty": {}, "explicit-no-aux": execution, "old-writer": execution}.get(case, group)
    lost = None if case == "omitted" else {"body": {"execution": sent}}
    head = {"execution": group}
    if case == "old-writer":
        head.update(access="workspace_write", execution={"isolation": "live", "delegated": True})
    _queue, _workers, successor = _held_continue(tmp_path, monkeypatch, head=head, lost=lost)
    if case == "prior-body-unavailable":
        materialize = delegate_pending.request_body
        monkeypatch.setattr(delegate_pending, "request_body", lambda root, row, **kw: (
            None if row.get("invocation_id") == "inv-head" else materialize(root, row, **kw)))
    refusal = _resume("owner", selected=successor)
    assert refusal["error"] == "predecessor_writers_unsettled", refusal
    if case in ("omitted", "empty", "explicit", "explicit-no-aux"):
        blocker, = refusal["blockers"]
        assert blocker["read_only"] is True and refusal["acceptable_unknown"] == [blocker["choice_id"]]
    else:
        assert "acceptable_unknown" not in refusal and not any(b.get("choice_id") for b in refusal["blockers"])


@pytest.mark.parametrize("same_family", [True, False], ids=["recorded-continue", "different-work"])
def test_read_only_proof_reuses_the_owner_continue_binding(tmp_path, monkeypatch, same_family):
    from ouroboros.delegate_resume import _read_only_chain
    from ouroboros.task_results import STATUS_CANCELLED, write_task_result
    from supervisor.continuation_admission import admit_continuation

    _install_queue(tmp_path, monkeypatch)
    _running_predecessor(tmp_path, PRED)
    _head(tmp_path)
    write_task_result(tmp_path, PRED, STATUS_CANCELLED, result="interrupted", **_stop_record(PRED))
    admitted = admit_continuation(PRED, action_nonce=NONCE)
    assert admitted["ok"] and not admitted.get("held"), admitted
    starter = admitted["successor_task_id"] if same_family else "another-work"
    _running_predecessor(tmp_path, starter)
    _lost(tmp_path, starter=starter)
    assert _read_only_chain(tmp_path, starter, LOST) is same_family


@pytest.mark.parametrize("case", ["at-limit", "too-long", "cycle", "unreadable"])
def test_the_resume_proof_has_the_same_member_bound_as_admission(case):
    from ouroboros.delegate_resume import _chain
    from ouroboros.owner_continue import WORK_CHAIN_MAX_HOPS, binding_sha

    limit = WORK_CHAIN_MAX_HOPS
    roots = [f"episode-{i}" for i in range(limit + (case in {"too-long", "unreadable"}))]
    rows = {tid: {} for tid in roots}
    links = roots[1:] + ([roots[0]] if case == "cycle" else [])
    for predecessor, successor in zip(roots, links):
        binding = {"verb": "continue", "predecessor_task_id": predecessor,
                   "successor_task_id": successor, "action_nonce": "nonce-fixture"}
        identity = {"binding": binding, "binding_sha256": binding_sha(binding)}
        rows[predecessor]["continued_by"] = {**identity, "successor_task_id": successor, "state": "admitted"}
        rows[successor]["continuation_admission"] = identity
    selected = roots[-1] if case != "cycle" else "not-reached"
    if case == "unreadable":
        del rows[selected]
    found, evidence, gap = _chain(rows.get, roots[0], selected)
    if case == "at-limit":
        assert len(found) == 64 and found == roots and evidence == rows and not gap
    else:
        assert not found and not evidence
        assert gap == {"too-long": "episode_chain_depth_exceeded", "cycle": "episode_chain_cycle",
                       "unreadable": "episode_record_unavailable"}[case]


@pytest.mark.parametrize("door", ["writer_hold", "owner_pause"])
def test_the_next_continue_names_the_carried_start_afresh_and_its_late_activity_holds_it(tmp_path, monkeypatch, door):
    """The successor named the start, ran and a Restart cut it; its own Continue inherits no decision. Only that
    Continue's own addressed Resume, through the verified chain and the successor's recorded choice, names the start
    again: a new exact choice bound to its own stop, never sent again. The old run's activity then holds it, and the
    run's money stays with the old tree."""
    from ouroboros import owner_pause
    from ouroboros.delegate_custody_current import run
    from ouroboros.task_results import STATUS_CANCELLED, load_task_result, write_task_result
    from supervisor.continuation_admission import admit_continuation, conflicting_writers
    from supervisor.owner_pause_control import request_owner_pause
    from tests.test_resume_attempt_history import _work

    queue, workers, successor = _held_continue(tmp_path, monkeypatch)
    choice, = _resume("owner", selected=successor)["acceptable_unknown"]
    assert _resume("owner", [choice], selected=successor)["ok"]
    workers.PENDING[:] = [row for row in workers.PENDING if row["id"] != successor]  # it ran, then a Restart cut it
    write_task_result(tmp_path, successor, STATUS_CANCELLED, result="interrupted",
                      **_stop_record(successor, request_id="ci_6a7b8c9d0e1f"))
    ack = admit_continuation(successor, action_nonce="press-0002-abcdef")
    assert ack["ok"] and ack["held"], ack
    second = ack["successor_task_id"]
    assert set(_decisions(tmp_path)) == {successor}, "no decision passes to the next episode by itself"
    work = _work(tmp_path, second)
    assert (work["activity"], work["because"][0]["code"]) == ("waiting", "earlier_work_unconfirmed")
    assert {"action": "resume", "target": second} in work["controls"], "the carried start is still owned by PRED"
    listed = _list_row(tmp_path, second)
    assert listed.get("ok") is False and not listed.get("acceptable_unknown"), listed
    fence_id = ""
    if door == "owner_pause":
        by = {"actor": "owner", "text": "", "at": "2026-11-25T00:00:00+00:00"}
        assert request_owner_pause(second, request_id="pause-second", by=by)["ok"]
        fence_id = owner_pause.read_fence(tmp_path, second)["fence_id"]
    refusal = _resume("owner", selected=second)
    again, = refusal["acceptable_unknown"]
    assert again.startswith(f"delegate_start:{PRED}/{PRED}/{LOST}/{quote(second, safe='')}/{fence_id}/"), again
    assert refusal["blockers"][0]["read_only"] is True

    ok = _resume("owner", [again], selected=second)
    assert ok["ok"], ok
    risk, = ok["accepted_start_risks"]
    assert (risk["successor_of"], risk["resuming_task_id"], risk["creation"]) == (PRED, second, "lookup_only")
    assert set(_decisions(tmp_path)) == {successor, second} and len(_rows(tmp_path, dc.START_RISK_ACCEPTED)) == 2
    assert load_task_result(tmp_path, second)["accepted_predecessor_starts"] == {
        LOST: {"root_task_id": PRED, "task_id": PRED, "invocation_id": LOST}}
    assert dc.invocation_record(tmp_path, LOST)["state"] == "pending", "the old request is only looked up"

    assert _bind_late(tmp_path).durable
    fence = owner_pause.read_fence(tmp_path, second)
    assert owner_pause.fence_closed(fence) and fence["active_delegated_start"]["holder_task_id"] == second, fence
    assert not owner_pause.fence_closed(owner_pause.read_fence(tmp_path, successor)), \
        "the Restart's own stop owns what the cut successor left"
    assert any(b.get("run_id") == "run-late" for b in conflicting_writers(queue, second, drive_root=tmp_path))
    late = run(tmp_path, "run-late")
    assert (late.task_id, late.root_task_id) == (PRED, PRED)
    assert late.billing_group.get("billing_group_id") not in {successor, second}, late.billing_group


@pytest.mark.parametrize("tail", ["child", "review", "upkeep", "closed-absent", "none"])
def test_an_answered_successor_is_held_again_only_while_its_tree_still_runs(tmp_path, monkeypatch, tail):
    """Late activity holds unanswered ownership of the successor, not merely a persisted run id."""
    from ouroboros import owner_pause, review_operation
    from ouroboros.task_results import STATUS_COMPLETED, STATUS_RUNNING, write_task_result

    _queue, workers, successor = _held_continue(tmp_path, monkeypatch)
    choice, = _resume("owner", selected=successor)["acceptable_unknown"]
    assert _resume("owner", [choice], selected=successor)["ok"]
    workers.PENDING[:] = [row for row in workers.PENDING if row["id"] != successor]
    upkeep = {"root_phase_checkpoint": {"post_task_synthesis": "pending_once"}} if tail == "upkeep" else {}
    write_task_result(tmp_path, successor, STATUS_COMPLETED, result="The report is ready.", **upkeep)
    if tail == "child":
        write_task_result(tmp_path, "child-1", STATUS_RUNNING, root_task_id=successor, parent_task_id=successor)
    elif tail == "review":
        review_operation._update_operations(tmp_path, successor, lambda rows: {
            **rows, "review-fixture": {"state": review_operation.OPERATION_PREPARING}}, create=True)
        monkeypatch.setattr(review_operation, "controller_state",
                            lambda *_: pytest.fail("late custody write must not probe a controller"))
    elif tail == "closed-absent":
        assert dc.record_started(tmp_path, dc.RunCustody(
            run_id="run-absent", task_id=successor, root_task_id=successor, route_id="codex", model="m",
            ledger_root=str(tmp_path)))
        assert dc.emit(tmp_path, dc.CLOSED_ABSENT, {"run_id": "run-absent", "task_id": successor})
    assert _bind_late(tmp_path).durable
    fence = owner_pause.read_fence(tmp_path, successor)
    live = tail in {"child", "review", "upkeep"}
    assert owner_pause.fence_closed(fence) is live, fence
    if live:
        assert fence["observe_only"] and fence["active_delegated_start"]["holder_task_id"] == successor
    if tail == "child":
        ctx, _limit = _loop_ctx(tmp_path, "child-1")
        ctx.root_task_id = successor  # the child's own launch gate reads its tree's fence
        with pytest.raises(owner_pause.OwnerPauseRefused):
            owner_pause.run_operation(ctx, lambda: pytest.fail("the child launched past the old run's activity"))


def test_unread_or_busy_successors_leave_the_activity_owed_and_are_never_waited_for(tmp_path, monkeypatch):
    """Successors that cannot be read under the old tree's lock are unknown, never none; a successor's busy lock is
    tried, never waited for (a Stop's or Panic's own read-back publishes here). Either way nothing lands and the
    proof of activity stays owed, until a later one lands it with the successor's lock in the same order."""
    from ouroboros import delegate_resume, owner_pause, store

    _queue, _workers, successor = _held_continue(tmp_path, monkeypatch)
    choice, = _resume("owner", selected=successor)["acceptable_unknown"]
    assert _resume("owner", [choice], selected=successor)["ok"]
    real_holders, real_lock, taken, state = delegate_resume.resume_holders, owner_pause.launch_lock, [], {}

    def holders(record, tree):
        if state.pop("unread", False):
            raise store.StoreBusy("held by another writer", reason="contention")
        return real_holders(record, tree)

    @contextlib.contextmanager
    def lock(root, root_task_id, **kwargs):
        taken.append((str(root_task_id), kwargs.get("wait", True)))
        if state.get("busy") and root_task_id == successor:
            raise owner_pause.LaunchBusy("contention")
        with real_lock(root, root_task_id, **kwargs):
            yield

    def owed():
        return (dc.invocation_record(tmp_path, LOST)["state"] == "pending"
                and not owner_pause.fence_closed(owner_pause.read_fence(tmp_path, successor)))

    monkeypatch.setattr(delegate_resume, "resume_holders", holders)
    monkeypatch.setattr(owner_pause, "launch_lock", lock)
    state["unread"] = True
    assert not _bind_late(tmp_path).durable and owed()
    assert successor not in [name for name, _wait in taken], "no successor lock is taken while they are unread"
    state["busy"], taken[:] = True, []
    dc._CUSTODY.clear()
    assert not _bind_late(tmp_path).durable and owed()
    assert (successor, False) in taken, taken
    state["busy"] = False
    dc._CUSTODY.clear()
    assert _bind_late(tmp_path).durable
    assert owner_pause.fence_closed(owner_pause.read_fence(tmp_path, successor))


def test_a_failed_successor_read_orders_the_deferred_fence_discovery(tmp_path, monkeypatch):
    """The body wakes each recovered start's tree and its successors from one read per listing; a read that fails still
    wakes the trees and orders the existing deferred discovery, which finds a successor's fence by its holder, so a
    successor is never lost to the body's memo of an unchanged row (``BodyRuntime._held``)."""
    from ouroboros import delegate_resume, store
    from supervisor import owner_pause_control, queue

    deferred, woken = [], []
    monkeypatch.setattr(queue, "owns_live_queue", lambda _root: True)
    monkeypatch.setattr(owner_pause_control, "defer_failed_recovery_discovery", deferred.append)
    monkeypatch.setattr(delegate_resume, "recovered_writer_changed", lambda _root, tree: woken.append(tree) or True)

    @contextlib.contextmanager
    def unreadable(*_args, **_kwargs):
        raise store.StoreBusy("held by another writer", reason="contention")
        yield  # never reached: a store read that never opens

    monkeypatch.setattr(store, "open_store", unreadable)
    entries = [dc.RunCustody(run_id=f"run-{n}", task_id=PRED, root_task_id=PRED, invocation_id=f"inv-{n}",
                             route_id="codex", model="m", ledger_root=str(tmp_path)) for n in (1, 2)]
    delegate_resume.recovered_writers_changed(tmp_path, entries)
    assert woken == [PRED] and deferred == [queue.DRIVE_ROOT], (woken, deferred)


def test_the_census_reads_only_the_carried_invocation_ids(tmp_path, monkeypatch):
    """Read only the two addressed invocations, not every run their starter ever owned."""
    from ouroboros import delegate_invocation, delegate_pending
    from ouroboros.delegate_resume import carried_start_writers

    _running_predecessor(tmp_path, PRED)
    _head(tmp_path)
    _lost(tmp_path)
    for iid in ("inv-b", "inv-other"):
        assert dc.record_start_requested(tmp_path, task_id=PRED, root_task_id=PRED, invocation_id=iid,
                                         idempotency_key=iid, request=BODY, route="codex")
    assert dc.record_started(tmp_path, dc.RunCustody(run_id="run-b", task_id=PRED, root_task_id=PRED,
                                                     invocation_id="inv-b", route_id="codex", model="m",
                                                     ledger_root=str(tmp_path)))
    reads, bound_runs = [], delegate_invocation.bound_runs

    def observed(txn, iid):
        reads.append(iid)
        return bound_runs(txn, iid)

    monkeypatch.setattr(delegate_invocation, "bound_runs", observed)
    monkeypatch.setattr(delegate_pending, "request_body", lambda *_: pytest.fail("liveness needs no body"))
    carried = {iid: {"root_task_id": PRED, "task_id": PRED, "invocation_id": iid} for iid in (LOST, "inv-b")}
    rows = carried_start_writers(tmp_path, carried, unbound=True)
    assert reads == sorted(carried) and {r["invocation_id"]: r["run_id"] for r in rows} == {LOST: "", "inv-b": "run-b"}
    assert [row["run_id"] for row in carried_start_writers(tmp_path, carried, unbound=False)] == ["run-b"]
    carried["inv-b"]["task_id"] = "wrong-task"
    assert carried_start_writers(tmp_path, carried, unbound=False) == [
        {"kind": "custody_unreadable", "task_id": "wrong-task", "invocation_id": "inv-b"}]
