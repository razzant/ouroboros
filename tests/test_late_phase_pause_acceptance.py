"""Owner D10: Pause of an answered root while its retained late acceptance is live.

The real historical review owner, its retained operation and settlement, the
owner's Pause ingress, the assignment tick's Pause census and the Resume seam.
The reviewer transport is the shared synthetic one (``late`` fixture): it still
reserves, dispatches and settles every send through the usage ledger, so the
owner fence gates new sends exactly as in production.
"""

from __future__ import annotations

import queue
import json
import subprocess
import sys
import threading
import time

import pytest

from ouroboros import review_operation
from ouroboros.task_results import load_task_result
from supervisor import events_chat_delivery as chat
from tests.test_acceptance_history import _caller, _request, _source
from tests.test_acceptance_late_consumers import delivered
from tests.test_acceptance_late_consumers import late as late  # noqa: F401 — fixture
from tests.test_review_operation_collection import _send_ctx
from tests.test_review_operation_collection import fresh_sends as fresh_sends  # noqa: F401 — fixture
from tests.test_review_operation_lifetime import until


def _census(root) -> dict:
    from ouroboros.gateway.state import _chat_activities_snapshot_safe

    return {row["activity_id"]: row["phase"] for row in _chat_activities_snapshot_safe(root, {}, direct_turns=[])}


def _tick(q):
    """The assignment tick's observe-only Pause census (its per-root throttle reset)."""
    from supervisor import owner_pause_control

    owner_pause_control._LAST_SETTLE_CHECK.clear()
    return owner_pause_control.settle_requested_owner_pauses(q)


@pytest.mark.parametrize("registered", [False, True])
def test_pause_during_sent_late_acceptance_collects_it_once_and_releases_when_nothing_remains(
        late, tmp_path, monkeypatch, registered):  # noqa: F811
    """Pause is accepted after delivery (with and without a RUNNING row); the already
    dispatched panel finishes and is collected once — never re-sent — and the delivered
    answer stays exactly as it was. Owner 2026-10-08 (full variant): with no writer left
    the tree reads Paused while its launched reviewers finish separately; a live writer
    keeps it Pausing. Nothing was left to defer, so the Pause releases itself."""
    from ouroboros.owner_pause import read_fence
    from supervisor.owner_pause_control import request_owner_pause
    from tests._budget_pause_exact_helpers import _install_queue

    f = delivered(tmp_path, monkeypatch)
    q, _state, workers = _install_queue(f.root, monkeypatch)
    ctx = _caller(f)
    ctx.event_queue = queue.Queue()
    release = threading.Event()
    late.gates.append(release)
    before = load_task_result(f.root, f.tid)
    try:
        start = _request(f, ctx, _source(ctx, text="Review this delivered historical answer."))
        assert start["status"] == "pending", start
        until(lambda: len(late.calls) == 3)
        if registered:
            workers.RUNNING[f.tid] = {"task": f.task, "attempt": 1, "worker_id": 0}
        pause = request_owner_pause(f.tid, request_id="late-acceptance-pause")
        expected = "requested" if registered else "paused"
        assert pause["ok"] and pause["state"] == expected, pause
        assert _census(f.root)[f.tid] == ("budget_pausing" if registered else "budget_paused")
        _tick(q)
        fence = read_fence(f.root, f.tid)
        assert fence["state"] == expected  # a live writer keeps Pausing; reviewers alone never do
        if not registered:
            assert fence["finishing_reviews"], fence  # still finishing, shown separately
    finally:
        workers.RUNNING.clear()
        release.set()
    until(lambda: not review_operation._LIVE)
    after = load_task_result(f.root, f.tid)
    assert after["result"] == before["result"] and after["status"] == before["status"]
    assert len(late.calls) == 3  # the dispatched panel was collected, never reissued
    panels = (after.get("review_projection") or {}).get("panels") or []
    assert panels and all(actor.get("operation_state") != "not_dispatched"
                          for panel in panels for actor in panel.get("actors") or [])
    _tick(q)
    assert read_fence(f.root, f.tid)["state"] == "released" and f.tid not in q.BUDGET_ROOT_FENCES
    assert f.tid not in _census(f.root)
    # The same frozen subject keeps its single paid identity: a new owner request only collects.
    again = _request(f, ctx, _source(ctx, text="Review the same delivered answer again."))
    until(lambda: not review_operation._LIVE)
    assert again["status"] != "pending" and len(late.calls) == 3, again


def test_pause_defers_an_unsent_automatic_late_review_until_resume(late, tmp_path, monkeypatch):  # noqa: F811
    """An automatic late review still preparing (nothing sent) is deferred by the Pause —
    not cancelled — shows as Paused, and runs exactly once after the owner's Resume."""
    from ouroboros.owner_pause import read_fence
    from supervisor.owner_pause_control import request_owner_pause
    from tests._budget_pause_exact_helpers import _install_queue

    f = delivered(tmp_path, monkeypatch, receipt="owed")
    q, _state, workers = _install_queue(f.root, monkeypatch)
    workers.RUNNING[f.tid] = {"task": f.task, "attempt": 1, "worker_id": 0}  # its writer still runs
    chat._handle_send_message(f.event, _send_ctx(f.root, []))
    until(lambda: bool(review_operation._LIVE))
    operation = next(iter(review_operation._LIVE.values()))
    pause = request_owner_pause(f.tid, request_id="defer-automatic-review")
    assert pause["ok"], pause
    workers.RUNNING.clear()  # the original writer ends while the Pause stands
    time.sleep(0.6)  # several preparation polls and control re-reads
    assert operation.control() is None and not operation.closed and not late.calls
    pointer = next(iter(load_task_result(f.root, f.tid)["review_operations"].values()))
    assert pointer["state"] == "preparing"
    _tick(q)
    assert read_fence(f.root, f.tid)["state"] == "paused"  # nothing sent is in flight
    assert _census(f.root)[f.tid] == "budget_paused"
    resumed = q.resume_budget_paused_task(f.tid)
    assert resumed["ok"] and resumed["owner_pause_released"], resumed
    until(lambda: len(late.calls) == 3)
    until(lambda: not review_operation._LIVE)
    assert len(late.calls) == 3 and read_fence(f.root, f.tid)["state"] == "released"


_PREPARER = '''
import json, sys, threading, time
from pathlib import Path
from types import SimpleNamespace
from ouroboros import acceptance_late, review_operation
from ouroboros.task_results import load_task_result
from ouroboros.llm import LLMClient
root, task_id, output = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
acceptance_late._historical_writer_live = lambda *_a, **_kw: True
review_operation._CONTROL_RECHECK_SEC = 3600.0
# Retaining the pointer does not mean the worker has finished its first control
# read. Publish readiness only after that read, before the parent installs a rail.
ready = threading.Event()
control = review_operation.ReviewOperation.control
def checked_control(self):
    result = control(self)
    if not ready.is_set():
        assert result is None, result
        ready.set()
    return result
review_operation.ReviewOperation.control = checked_control
LLMClient.chat = lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("unexpected paid send"))
ctx = SimpleNamespace(task_id=task_id, task_attempt=1, drive_root=root, budget_drive_root=root,
    task_metadata={"root_task_id": task_id, "budget_drive_root": str(root)}, event_queue=None, pending_events=[])
row = load_task_result(root, task_id, strict=True)
result = acceptance_late.run_historical_acceptance(ctx, task_id=task_id,
    debt_id=row["acceptance_debt"]["debt_id"], automatic=True)
assert ready.wait(10), 'preparation worker did not reach its control wait'
temporary = output.with_suffix('.tmp')
temporary.write_text(json.dumps(result), encoding="utf-8")
temporary.replace(output)
while True: time.sleep(1)
'''


def _preparing_process(f):
    """Real retained controller; only its original-writer wait is a fixture."""
    result, path = f.root / 'preparer.json', f.root / 'preparer.log'
    log = path.open('w', encoding='utf-8')
    proc = subprocess.Popen([sys.executable, '-c', _PREPARER, str(f.root), f.tid, str(result)],
                            stdout=log, stderr=subprocess.STDOUT)
    try:
        until(lambda: result.exists() or proc.poll() is not None)
        assert proc.poll() is None, path.read_text(encoding='utf-8')
        start = json.loads(result.read_text(encoding='utf-8'))
        assert start['status'] == 'preparing' and start['dispatched'] is False, start
        return proc, log, start
    except BaseException:
        _end_preparer(proc, log)
        raise


def _end_preparer(proc, log):
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=5)
    log.close()


def test_paused_preparation_survives_dead_controller_and_resumes_same_debt_once(late, tmp_path, monkeypatch):  # noqa: F811
    from ouroboros.owner_pause import read_fence
    from tests._budget_pause_exact_helpers import _install_queue
    from supervisor.owner_pause_control import request_owner_pause

    f = delivered(tmp_path, monkeypatch)
    q, _, _ = _install_queue(f.root, monkeypatch)
    proc, log, start = _preparing_process(f)
    try:
        assert request_owner_pause(f.tid, request_id='preparation-restart')['ok']
        _tick(q)
        before = load_task_result(f.root, f.tid)
        entry = before['review_operations'][start['owner_id']]
        assert entry['preparation_pause']['intent_ref'] == entry['intent_ref']
        assert review_operation.controller_state(entry['controller']) == 'alive'
    finally:
        _end_preparer(proc, log)
    assert review_operation.controller_state(entry['controller']) == 'dead'
    q.BUDGET_ROOT_FENCES.clear()  # The new server starts without process-local latches.
    for _ in range(2):
        report = review_operation.recover_orphaned_acceptance_operations(f.root)
        assert report['deferred'][0]['reason'] == 'owner_paused_preparation', report
        _tick(q)
        assert read_fence(f.root, f.tid)['state'] == 'paused'
        assert q.BUDGET_ROOT_FENCES[f.tid]['cause'] == 'owner_pause'
        assert _census(f.root)[f.tid] == 'budget_paused'
        assert not review_operation.task_has_live_review_operation(f.root, f.tid)
        assert not late.calls  # Maintenance never buys or recreates a live owner.
    resumed = q.resume_budget_paused_task(f.tid)
    assert resumed['ok'], resumed
    q.resume_budget_paused_task(f.tid)  # Repeated owner input cannot start another controller.
    until(lambda: len(late.calls) == 3)
    until(lambda: not review_operation._LIVE)
    after = load_task_result(f.root, f.tid)
    assert set(after['review_operations']) == {start['owner_id']}
    assert all(scope.task_id == f.tid and scope.root_task_id == f.accounting and scope.root_limit_usd == 4.0
               for scope, _ in late.calls)
    assert after['acceptance_debt'] == before['acceptance_debt']
    assert after['status'] == before['status'] and after['result'] == before['result']
    review_operation.recover_orphaned_acceptance_operations(f.root)
    q.resume_budget_paused_task(f.tid)
    assert len(late.calls) == 3


@pytest.mark.parametrize('bound', ['global', 'root', 'deadline', 'restart', 'panic', 'cancel'])
def test_preparing_acceptance_resume_keeps_pause_under_existing_rails(late, tmp_path, monkeypatch, bound):  # noqa: F811
    from ouroboros.owner_pause import read_fence
    from ouroboros.task_results import write_task_result
    from ouroboros.usage_accounting import AttemptRequest, execute_physical_attempt, read_usage_records
    from supervisor import budget_resume, state
    from supervisor.owner_pause_control import request_owner_pause
    from tests._budget_pause_exact_helpers import _install_queue

    f = delivered(tmp_path, monkeypatch)
    q, _, _ = _install_queue(f.root, monkeypatch)
    proc, log, _ = _preparing_process(f)
    try:
        if bound in {'global', 'root'}:
            execute_physical_attempt(AttemptRequest(model='openai/gpt-4.1-nano', provider='openai',
                drive_root=f.root, task_id=f.tid, root_task_id=f.tid, root_limit_usd=4.0, reservation_usd=4.0),
                lambda: 'prior cost', extractor=lambda _: ({}, 4.0, True))
        assert request_owner_pause(f.tid, request_id='prepare-' + bound)['ok']
        _tick(q)
        before = read_fence(f.root, f.tid)
        assert before['state'] == 'paused'
        if bound == 'global':
            monkeypatch.setattr(state, 'TOTAL_BUDGET_LIMIT', 4.0)
            assert budget_resume._global_money_refusal(q, state.budget_remaining)['error'] == 'budget_still_exhausted'
        elif bound == 'root':
            row = {**load_task_result(f.root, f.tid), 'id': f.tid}
            assert budget_resume._root_money_refusal(f.root, row, f.tid)['error'] == 'root_hard_cap_exhausted'
        elif bound == 'deadline':
            write_task_result(f.root, f.tid, 'completed', deadline_at='2000-01-01T00:00:00Z')
        elif bound in {'restart', 'panic'}:
            flag = 'owner_restart_no_resume.flag' if bound == 'restart' else 'panic_stop.flag'
            (f.root / 'state' / flag).write_text(bound, encoding='utf-8')
        else:
            from ouroboros.cancel_intents import request_cancel
            request_cancel(f.root, f.tid, source='owner', reason='Stop', allow_settled_target=True)
        money = read_usage_records(f.root)
        result = q.resume_budget_paused_task(f.tid)
        assert not result['ok'], result
        assert read_fence(f.root, f.tid) == before
        assert not late.calls and read_usage_records(f.root) == money
    finally:
        _end_preparer(proc, log)


@pytest.mark.parametrize('damage', ['unknown', 'source', 'generation'])
def test_dead_preparation_needs_its_exact_pause_and_source(late, tmp_path, monkeypatch, damage):  # noqa: F811
    from ouroboros.owner_pause import read_fence
    from ouroboros.task_results import write_task_result
    from supervisor.owner_pause_control import request_owner_pause
    from tests._budget_pause_exact_helpers import _install_queue

    f = delivered(tmp_path, monkeypatch)
    q, _, _ = _install_queue(f.root, monkeypatch)
    proc, log, start = _preparing_process(f)
    try:
        assert request_owner_pause(f.tid, request_id='bound-source')['ok']
    finally:
        _end_preparer(proc, log)
    row = load_task_result(f.root, f.tid)
    entry = row['review_operations'][start['owner_id']]
    if damage == 'unknown':
        entry['state'] = 'preparation_unknown'
    elif damage == 'source':
        from ouroboros.artifacts import task_artifact_dir_path
        path = task_artifact_dir_path(f.root, f.tid) / entry['intent_ref']['path']
        path.write_text('{"not_the_saved_intent":true}', encoding='utf-8')
    else:
        entry['preparation_pause']['generation'] += 1
    write_task_result(f.root, f.tid, row['status'], review_operations=row['review_operations'])
    _tick(q)
    before = read_fence(f.root, f.tid)
    assert before['state'] == 'paused'
    refused = q.resume_budget_paused_task(f.tid)
    assert not refused['ok'], refused
    assert read_fence(f.root, f.tid) == before and not late.calls
    assert load_task_result(f.root, f.tid)['review_operations'] == row['review_operations']


def test_owner_stop_reaches_dead_paused_preparation_through_real_http(late, tmp_path, monkeypatch):  # noqa: F811
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient
    from ouroboros.gateway import tasks
    from ouroboros.owner_pause import read_fence
    from supervisor.owner_pause_control import request_owner_pause
    from tests._budget_pause_exact_helpers import _install_queue

    f = delivered(tmp_path, monkeypatch)
    q, _, _ = _install_queue(f.root, monkeypatch)
    proc, log, start = _preparing_process(f)
    try:
        assert request_owner_pause(f.tid, request_id='stop-saved-review')['ok']
    finally:
        _end_preparer(proc, log)
    review_operation.recover_orphaned_acceptance_operations(f.root)
    _tick(q)
    before = load_task_result(f.root, f.tid)
    app = Starlette(routes=[Route('/api/tasks/{task_id}/cancel', tasks.api_task_cancel, methods=['POST'])])
    app.state.drive_root = f.root
    with TestClient(app) as client:
        response = client.post(f'/api/tasks/{f.tid}/cancel', json={})
    assert response.status_code == 200, response.text
    after = load_task_result(f.root, f.tid)
    assert after['review_operations'][start['owner_id']]['preparation_outcome']['reason'] == 'owner_stopped'
    assert after['result'] == before['result'] and after['status'] == before['status']
    assert read_fence(f.root, f.tid)['state'] == 'released'
    assert f.tid not in _census(f.root) and not late.calls
    assert not q.resume_budget_paused_task(f.tid)['ok']


def test_typed_preparation_refusal_does_not_leave_a_paused_remainder(late, tmp_path, monkeypatch):  # noqa: F811
    from ouroboros.owner_pause import read_fence
    from ouroboros.task_results import write_task_result
    from supervisor.owner_pause_control import request_owner_pause
    from tests._budget_pause_exact_helpers import _install_queue

    f = delivered(tmp_path, monkeypatch)
    q, _, workers = _install_queue(f.root, monkeypatch)
    workers.RUNNING[f.tid] = {'task': f.task, 'attempt': 1, 'worker_id': 0}
    from ouroboros.acceptance_late import run_historical_acceptance
    ctx = _caller(f)
    run_historical_acceptance(ctx, task_id=f.tid,
        debt_id=load_task_result(f.root, f.tid)['acceptance_debt']['debt_id'], automatic=True)
    until(lambda: bool(review_operation._LIVE))
    assert request_owner_pause(f.tid, request_id='preparation-refusal')['ok']
    # A real typed control refusal ends unsent work instead of leaving an obligation.
    write_task_result(f.root, f.tid, 'completed', deadline_at='2000-01-01T00:00:00Z')
    workers.RUNNING.clear()
    until(lambda: not review_operation._LIVE)
    _tick(q)
    pointer = next(iter(load_task_result(f.root, f.tid)['review_operations'].values()))
    assert pointer['state'] == 'preparation_refused', pointer
    assert read_fence(f.root, f.tid)['state'] == 'released'
    assert f.tid not in _census(f.root) and not late.calls
