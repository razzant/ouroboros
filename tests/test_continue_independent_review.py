"""#1572: the owner's Continue is not held by an independent review that cannot
apply to the candidate; every other census consumer keeps the whole observation.

Real custody rows (``delegate_custody``), the real observer (``budget_pause.
observe_task_runs``), the real writer census (``conflicting_writers``) and
both real Continue consumers (admission and held selection) run here. The
independent-review predicate is the shared ``delegate_registration_policy``
helper (builder A's owner); no transport, reviewer or model is involved.
"""
from __future__ import annotations

import os

import pytest

from ouroboros import delegate_custody as dc
from tests._budget_pause_exact_helpers import _install_queue
from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool
from tests.test_owner_continue import NONCE, _interrupted
from tests.worker_pool_support import affirm_ready

pytestmark = pytest.mark.serial

PRED = "pred-1"
SLOT = "review-session"


@pytest.fixture(autouse=True)
def _current_pool(monkeypatch):
    from ouroboros.reviewer_slot_config import review_pool_rows

    set_review_pool(monkeypatch, pool_roster(pool_seat(
        SLOT, "codex=gpt-5.2", kind="agent_session", delivery=None)))
    assert review_pool_rows()[0].slot_id == SLOT


def _rows(root, *, leaf=True, review=True, review_with_snapshot=False, review_investigation=False,
          pending_review=False):
    """The predecessor's live custody: an ordinary delegated leaf, a reviewer
    seat (readonly today; a private investigation copy under #1609), and the
    conservative legacy shapes."""
    dc._CUSTODY.clear()
    if leaf:
        assert dc.record_started(root, dc.RunCustody(
            run_id="run-leaf", task_id=PRED, route_id="codex", model="leaf-model",
            snapshot_id="snap-leaf", authority_source="active_workspace", ledger_root=str(root)))
    if review:
        # The read-only seat's own STARTED row (review_execution): no authority fact,
        # its granted readonly Ask shape on the same row.
        assert dc.record_started(root, dc.RunCustody(
            run_id="run-review", task_id=PRED, route_id="codex", model="review-model",
            source="review_substrate", category="task_acceptance_review",
            review_slot_id=SLOT, ledger_root=str(root)), shape=READONLY_SEAT_SHAPE)
    if review_with_snapshot:
        # A review-owned row whose recorded binding could still apply: held (uncertainty is not a grant).
        assert dc.record_started(root, dc.RunCustody(
            run_id="run-review-snapshot", task_id=PRED, route_id="codex", model="review-model",
            source="review_substrate:review_change", snapshot_id="snap-review", ledger_root=str(root)))
    if review_investigation:
        assert dc.record_started(root, dc.RunCustody(
            run_id="run-review-copy", task_id=PRED, route_id="codex", model="review-model",
            source="review_substrate:task_acceptance", snapshot_id="snap-copy",
            authority_source="investigation", ledger_root=str(root)))
    if pending_review:
        assert dc.record_start_requested(
            root, run_id="", task_id=PRED, invocation_id="inv-review", idempotency_key="inv-review",
            request=_readonly_seat_request(root), route="codex", source="review_substrate.extraction")
    dc._CUSTODY.clear()


READONLY_SEAT_SHAPE = {"access": "readonly", "mode": "ask", "isolation": "", "delegated": False}


def _readonly_seat_request(root, **changes):
    """The exact wire body an explicitly read-only reviewer seat stores before its POST."""
    from types import SimpleNamespace

    from ouroboros.review_session_preparation import prepare_review_session_request

    invocation = SimpleNamespace(timeout_sec=60, instructions="review", use_thread=False, output_schema=None,
                                 access="readonly", source_delivery=None)
    route = SimpleNamespace(route_id="codex", model="review-model", effort="", profile_id="")
    request = prepare_review_session_request(invocation, route, prompt="review packet", root=str(root),
                                             thread_id="", schema_asked=False)
    assert (request["access"], request["mode"], "execution" in request) == ("readonly", "ask", False)
    return {**request, **changes}


def _delegated(blockers):
    return sorted(b.get("run_id") or b.get("invocation_id") for b in blockers if b["kind"] == "delegated_run")


def test_the_observation_carries_source_and_independence_without_filtering(tmp_path, monkeypatch):
    from ouroboros import delegate_pending
    from ouroboros.budget_pause import observe_task_runs

    _rows(tmp_path, review_with_snapshot=True, review_investigation=True, pending_review=True)
    monkeypatch.setattr(delegate_pending, "request_body", lambda *a: pytest.fail("census opened a request body"))
    observed = observe_task_runs(tmp_path, PRED, reason="continuation_writer_check", request_stop=False)
    assert observed["custody_read"] == "ok"
    by_id = {row["run_id"] or row["invocation_id"]: row for row in observed["runs"]}
    assert set(by_id) == {"run-leaf", "run-review", "run-review-snapshot", "run-review-copy", "inv-review"}
    assert by_id["run-leaf"]["source"] == "delegated_subagent" and by_id["run-leaf"]["independent_review"] is False
    assert by_id["run-review"]["source"] == "review_substrate" and by_id["run-review"]["independent_review"] is True
    assert by_id["run-review-snapshot"]["independent_review"] is False
    assert by_id["run-review-copy"]["independent_review"] is True
    assert by_id["inv-review"]["source"] == "review_substrate.extraction"
    assert by_id["inv-review"]["independent_review"] is True
    assert by_id["inv-review"]["stop_outcome"] == "pending_invocation_unbound"


def test_only_the_continue_purpose_leaves_independent_reviews_out(tmp_path, monkeypatch):
    from supervisor.continuation_admission import CONTINUE_PURPOSE, conflicting_writers

    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, PRED)
    _rows(tmp_path, review_with_snapshot=True, review_investigation=True, pending_review=True)
    census = _delegated(conflicting_writers(queue, PRED))
    assert census == ["inv-review", "run-leaf", "run-review", "run-review-copy", "run-review-snapshot"]
    assert _delegated(conflicting_writers(queue, PRED, owner_pause_fence_id="fence-1")) == census
    assert _delegated(conflicting_writers(queue, PRED, purpose="")) == census
    assert _delegated(conflicting_writers(queue, PRED, purpose=CONTINUE_PURPOSE)) == ["run-leaf", "run-review-snapshot"]


def test_an_observation_without_the_independence_fact_keeps_holding(tmp_path, monkeypatch):
    from ouroboros import budget_pause
    from supervisor import continuation_admission
    from supervisor.continuation_admission import CONTINUE_PURPOSE, conflicting_writers

    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, PRED)
    _rows(tmp_path, leaf=False)
    real = budget_pause.observe_task_runs

    def older_projection(*args, **kwargs):
        observed = real(*args, **kwargs)
        for row in observed["runs"]:
            row.pop("independent_review", None)
            row.pop("source", None)
        return observed

    monkeypatch.setattr(continuation_admission, "observe_task_runs", older_projection, raising=False)
    monkeypatch.setattr(budget_pause, "observe_task_runs", older_projection)
    assert _delegated(conflicting_writers(queue, PRED, purpose=CONTINUE_PURPOSE)) == ["run-review"]


def _open_attempt(root, *, source, slot=None, review_skill="", request_source=None):
    """One API attempt still in flight through the real scope merge: the bound scope
    carries the slot attribution, the request may stamp its route's own ``source``.
    Its send has not returned, so it is a model request with no result for every census
    (an unbound send that already raised is not: tests/test_completed_model_handoff.py)."""
    from ouroboros import usage_accounting as ua

    slot = (SLOT if source.startswith("review_substrate") else "") if slot is None else slot
    scope = ua.UsageScope(drive_root=root, task_id=PRED, root_task_id=PRED, source=source,
                          **({"category": "task_acceptance_review", "review_slot_id": slot} if slot else {}),
                          review_skill=review_skill)
    with ua.usage_scope(scope):
        reservation = ua.reserve_attempt(ua.AttemptRequest(
            model="m", provider="test", reservation_usd=0.2, **({"source": request_source} if request_source else {})))
        ua.mark_dispatched(reservation, local_answer_owner_pid=os.getpid())
    row = ua.read_usage_records(root, final_only=True)[-1]
    assert row["state"] == "dispatched"
    assert (row["source"], row["review_slot_id"], row["review_skill"]) == (
        request_source or source, slot, review_skill)
    return row["attempt_id"]


def test_a_review_attributed_model_attempt_does_not_hold_continue_but_an_author_attempt_does(tmp_path, monkeypatch):
    from supervisor.continuation_admission import CONTINUE_PURPOSE, conflicting_writers

    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    _interrupted(tmp_path, PRED)
    dc._CUSTODY.clear()
    reviewer = _open_attempt(tmp_path, source="review_substrate")
    handoffs = lambda blockers: sorted(b["attempt_id"] for b in blockers if b["kind"] == "model_handoff")
    assert handoffs(conflicting_writers(queue, PRED)) == [reviewer]
    assert handoffs(conflicting_writers(queue, PRED, purpose=CONTINUE_PURPOSE)) == []
    author = _open_attempt(tmp_path, source="delegated_subagent")
    assert handoffs(conflicting_writers(queue, PRED)) == sorted([reviewer, author])
    assert handoffs(conflicting_writers(queue, PRED, purpose=CONTINUE_PURPOSE)) == [author]


def test_admission_and_held_selection_both_use_the_continue_purpose(tmp_path, monkeypatch):
    from supervisor.continuation_admission import admit_continuation
    from supervisor.events_budget import observe_held_budget_selection
    from ouroboros.work_stop import hold_fact as budget_hold_fact

    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, PRED, origin={"source": "owner_restart", "reason": "owner_restart", "scope": "single"})
    _rows(tmp_path, review_with_snapshot=False, pending_review=True)
    ack = admit_continuation(PRED, action_nonce=NONCE)
    assert ack["ok"] and ack["held"] is True, ack
    successor = ack["successor_task_id"]
    held = next(row for row in queue.PENDING if row["id"] == successor)
    hold = budget_hold_fact(held)
    assert hold["reason"] == "continuation_writer_unsettled"
    assert _delegated(hold["blockers"]) == ["run-leaf"], "only the author's own delegation holds"
    # The author's leaf settles; the reviewer seat and its unbound start stay live.
    dc._CUSTODY.clear()
    assert dc.emit(tmp_path, dc.SETTLED, {"run_id": "run-leaf", "task_id": PRED, "route": "codex",
                                          "model": "leaf-model", "state": "completed"})
    dc._CUSTODY.clear()
    observation = observe_held_budget_selection(queue, successor)
    assert "error" not in observation and observation["blockers"] == [], observation
    # The same census for the Pause/money readers still sees the reviewer seat.
    from supervisor.continuation_admission import conflicting_writers
    assert _delegated(conflicting_writers(queue, PRED)) == ["inv-review", "run-review"]


def test_continue_admits_unheld_when_only_independent_reviews_are_live(tmp_path, monkeypatch):
    from supervisor.continuation_admission import admit_continuation

    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, PRED, origin={"source": "owner_restart", "reason": "owner_restart", "scope": "single"})
    _rows(tmp_path, leaf=False, review_investigation=True, pending_review=True)
    ack = admit_continuation(PRED, action_nonce=NONCE)
    assert ack["ok"] and ack["held"] is False, ack
    successor = next(row for row in queue.PENDING if row["id"] == ack["successor_task_id"])
    assert successor["metadata"]["continuation"]["admission"]["held"] is False


def test_live_held_continue_resume_selects_same_successor_and_assigns_it(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ouroboros.work_stop import hold_fact as budget_hold_fact
    from supervisor.continuation_admission import admit_continuation, conflicting_writers

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, PRED, origin={"source": "owner_restart", "reason": "owner_restart", "scope": "single"})
    _rows(tmp_path, review_with_snapshot=False, pending_review=True)
    admitted = admit_continuation(PRED, action_nonce=NONCE)
    assert admitted["ok"] and admitted["held"] is True, admitted
    successor_id = admitted["successor_task_id"]
    successor = next(row for row in workers.PENDING if row["id"] == successor_id)
    assert budget_hold_fact(successor)["reason"] == "continuation_writer_unsettled"

    # The author's writer settles; the independent readonly review request remains
    # in custody and is still visible to Pause/money census readers.
    dc._CUSTODY.clear()
    assert dc.emit(tmp_path, dc.SETTLED, {"run_id": "run-leaf", "task_id": PRED,
                                          "route": "codex", "model": "leaf-model", "state": "completed"})
    dc._CUSTODY.clear()
    from supervisor.events_budget import observe_held_budget_selection
    observed = observe_held_budget_selection(queue, successor_id)
    assert observed["blockers"] == [] and observed["candidate"] is successor
    assert _delegated(conflicting_writers(queue, PRED)) == ["inv-review", "run-review"]

    # Public Resume consumes the held row's one selection. The successor identity
    # is unchanged and its exact continuation is what assignment consumes.
    resumed = queue.resume_budget_paused_task(successor_id)
    assert resumed["ok"] and resumed["selection"] == "budget_hold_released", resumed
    assert next(row for row in workers.PENDING if row["id"] == successor_id) is successor
    sent = []
    workers.WORKERS[0] = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                                         active_capacity=True, in_q=SimpleNamespace(put=lambda task: sent.append(dict(task)))))
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [successor_id]
    assert workers.RUNNING[successor_id]["task"]["id"] == successor_id
    assert sent[0]["metadata"]["continuation"]["predecessor_task_id"] == PRED
    billing = sent[0]["metadata"]["continuation"]
    assert billing["billing_group_id"] == PRED
    assert billing["billing_group_limit_usd"] == 20.0
    assert billing["billing_group_limit_source"] == "initial_task_admission"
    assert "_budget_pause_hold" not in sent[0] and "_resume_selection" in sent[0]
    assert sent[0]["_resume_selection"]["selected_by"] == "owner"
    # Resume released only this successor; the independent readonly request is
    # still unresolved custody for the money/Pause census.
    assert _delegated(conflicting_writers(queue, PRED)) == ["inv-review", "run-review"]


@pytest.mark.parametrize("source,authority,snapshot,resource", [
    ("delegated_subagent", "investigation", "copy", {}),
    ("unknown_review_source", "readonly", "", {}),
    ("review_substrate:task_acceptance", "unknown_authority", "", {}),
    ("review_substrate:review_change", "", "", {"workspace_kind": "directory", "strategy": "copy"}),
])
def test_unknown_or_author_owned_custody_still_holds_continue(tmp_path, monkeypatch, source, authority, snapshot, resource):
    from supervisor.continuation_admission import CONTINUE_PURPOSE, conflicting_writers

    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, PRED)
    _rows(tmp_path, leaf=False, review=False)
    assert dc.record_started(tmp_path, dc.RunCustody(
        run_id="possible-writer", task_id=PRED, route_id="codex", model="m", source=source,
        authority_source=authority, snapshot_id=snapshot, resource_ref=resource, ledger_root=str(tmp_path)))
    dc._CUSTODY.clear()
    assert _delegated(conflicting_writers(queue, PRED, purpose=CONTINUE_PURPOSE)) == ["possible-writer"]
    assert not dc.replay(tmp_path)["possible-writer"].settled


@pytest.mark.parametrize("scope_source,slot,review_skill,request_source,continue_holds", [
    # A reviewer slot's send on a route that keeps the scope's label (OpenAI-compatible, OpenRouter).
    ("review_substrate", SLOT, "", None, False),
    # The same slot on a route that stamps its own label on every send (llm_anthropic,
    # llm_gigachat, llm_local), or a reviewer's web search: still the reviewer's custody.
    ("review_substrate", SLOT, "", "llm.anthropic", False),
    ("review_substrate", SLOT, "", "web_search.openrouter", False),
    # Skill attribution is an independent producer binding even when no slot id
    # survives the adapter's source rewrite.
    ("review_substrate", "", "commit_triad", "llm.anthropic", False),
    # A review source with no slot/skill custody has no current producer: unknown, holds.
    ("review_substrate", "", "", None, True),
])
def test_api_attempt_attribution_cases(tmp_path, monkeypatch, scope_source, slot, review_skill, request_source,
                                      continue_holds):
    from supervisor.continuation_admission import CONTINUE_PURPOSE, conflicting_writers

    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    _interrupted(tmp_path, PRED)
    dc._CUSTODY.clear()
    attempt = _open_attempt(tmp_path, source=scope_source, slot=slot, review_skill=review_skill,
                            request_source=request_source)
    handoffs = lambda blockers: [b for b in blockers if b["kind"] == "model_handoff"]
    paused = handoffs(conflicting_writers(queue, PRED))
    assert [(b["attempt_id"], b["review_owned"]) for b in paused] == [
        (attempt, bool(slot or review_skill))], "Pause/Resume see it"
    assert [b["attempt_id"] for b in handoffs(conflicting_writers(queue, PRED, purpose=CONTINUE_PURPOSE))] == (
        [attempt] if continue_holds else [])


def _pending(root, invocation_id, *, source="review_substrate", request=None, **facts):
    assert dc.record_start_requested(
        root, run_id="", task_id=PRED, invocation_id=invocation_id, idempotency_key=invocation_id,
        request=_readonly_seat_request(root) if request is None else request, route="codex",
        **({"source": source} if source is not None else {}), **facts)


@pytest.mark.parametrize("case", [
    "missing_source", "unknown_source", "research_readonly", "snapshot_capture", "directory_copy",
    "shape_absent", "writable_without_authority", "execution_binding",
])
def test_an_unbound_start_without_positive_no_write_proof_holds_continue(tmp_path, monkeypatch, case):
    """Exact pending START_REQUESTED rows: only a review source over the stored readonly
    Ask request without capture is left out; arbitrary absent metadata proves nothing."""
    from supervisor.continuation_admission import CONTINUE_PURPOSE, conflicting_writers

    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, PRED)
    _rows(tmp_path, leaf=False, review=False)
    _pending(tmp_path, "inv-proven")
    variants = {
        "missing_source": dict(source=None),
        "unknown_source": dict(source="unknown_review_source"),
        "research_readonly": dict(source="delegated_subagent"),
        "snapshot_capture": dict(snapshot_id="snap-review"),
        "directory_copy": dict(resource_ref={"workspace_kind": "directory", "strategy": "copy"}),
        "shape_absent": dict(request={"prompt": "review packet"}),
        "writable_without_authority": dict(request=_readonly_seat_request(tmp_path, access="workspace_write",
                                                                         mode="agent")),
        "execution_binding": dict(request=_readonly_seat_request(
            tmp_path, execution={"isolation": "live", "delegated": True, "workspaceRoot": str(tmp_path / "x")})),
    }
    _pending(tmp_path, "inv-unproven", **variants[case])
    dc._CUSTODY.clear()
    assert _delegated(conflicting_writers(queue, PRED)) == ["inv-proven", "inv-unproven"]
    assert _delegated(conflicting_writers(queue, PRED, purpose=CONTINUE_PURPOSE)) == ["inv-unproven"]


def test_a_started_review_row_without_its_recorded_shape_holds_continue(tmp_path, monkeypatch):
    from supervisor.continuation_admission import CONTINUE_PURPOSE, conflicting_writers

    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, PRED)
    _rows(tmp_path, leaf=False, review=True)
    assert dc.record_started(tmp_path, dc.RunCustody(
        run_id="run-review-shapeless", task_id=PRED, route_id="codex", model="review-model",
        source="review_substrate", review_slot_id=SLOT, ledger_root=str(tmp_path)))
    dc._CUSTODY.clear()
    assert _delegated(conflicting_writers(queue, PRED, purpose=CONTINUE_PURPOSE)) == ["run-review-shapeless"]
