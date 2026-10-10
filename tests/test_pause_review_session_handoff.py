"""Pause/Resume at the real delegated review checkpoint, before its first POST.

Only the external gateway is controlled. The coordinator, executor, durable
invocation, author warm park, census and Resume grant are real consumers.
"""
from __future__ import annotations

import copy
import json
import threading
import time

import pytest

from ouroboros import delegate_custody, owner_pause, review_custody, review_pause
from ouroboros.review_execution import ReviewRouteKind
from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request
from ouroboros.task_results import load_task_result
from tests._review_session_route_shared import FakeGateway
from tests.test_review_operation_lifetime import TASK, _parent, env, until  # noqa: F401

pytestmark = pytest.mark.serial


@pytest.fixture
def session(env, monkeypatch, tmp_path, request):  # noqa: F811
    from ouroboros import claudexor_daemon, review_operation
    from ouroboros.gateways import claudexor
    from ouroboros.owner_wait import direct_owner_wait
    from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx
    from tests.test_direct_chat_turn_owner_control import _live_chat_agent

    f = env
    f.repo = tmp_path / "candidate"
    f.repo.mkdir()
    _, state, _ = _install_queue(f.root, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 100.)
    agent = _live_chat_agent(monkeypatch, task_id=TASK)
    agent._task_started_ts = agent._last_activity_ts = time.time()
    ctx, f.limit = _loop_ctx(f.root, TASK, direct=True)
    ctx.__dict__.update(f.ctx.__dict__)
    ctx.owner_wait_callback = direct_owner_wait
    ctx.current_chat_id = 0
    ctx.task_started_at = time.time()
    f.ctx = ctx
    f.posts, f.checkpoints, f.reads, f.threads = [], [], [], []
    f.checkpoint_entered, f.checkpoint_release = threading.Event(), threading.Event()
    f.poll_entered, f.poll_release = threading.Event(), threading.Event()
    f.checkpoint_release.set()
    f.poll_release.set()
    FakeGateway.reset()
    monkeypatch.setattr(delegate_custody, "_CUSTODY", {})
    monkeypatch.setattr(review_custody, "_ACTIVE", {})
    monkeypatch.setattr(review_custody, "_NO_RESEND", {})

    class Gateway(FakeGateway):
        def start_run(self, body, *, idempotency_key=""):
            f.posts.append({"token": idempotency_key, "body": copy.deepcopy(body)})
            return super().start_run(body, idempotency_key=idempotency_key)

        def get_run(self, run_id, **kwargs):
            f.reads.append(run_id)
            f.poll_entered.set()
            assert f.poll_release.wait(10), "external observation barrier timed out"
            return super().get_run(run_id, **kwargs)

    def checkpoint(**facts):
        # The real executor's START_REQUESTED must already be durable here.
        record = delegate_custody.invocation_record(f.root, facts["invocation_id"])
        assert record["operation_id"] == facts["operation_id"]
        assert record["state"] == "pending"
        f.checkpoints.append(facts)
        f.checkpoint_entered.set()
        assert f.checkpoint_release.wait(10), "pre-POST checkpoint barrier timed out"

    f.ctx._review_pending_invocation_checkpoint = checkpoint
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", Gateway)
    monkeypatch.setattr(claudexor, "ClaudexorGateway", Gateway)
    monkeypatch.setattr(review_pause, "DETACH_POLL_SEC", .01)
    monkeypatch.setattr("ouroboros.review_execution._SESSION_POLL_SEC", .01)
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    yield f
    owner_pause.release_fence(f.root, TASK, reason="test_cleanup")
    f.checkpoint_release.set()
    f.poll_release.set()
    for thread in f.threads:
        thread.join(10)
        assert not thread.is_alive()
    until(lambda: not [op for op in review_operation._LIVE.values() if op.task_id == TASK])
    (f.root / "handoff_trace.json").write_text(json.dumps({
        "test": request.node.nodeid, "posts": f.posts, "checkpoints": f.checkpoints,
        "reads": f.reads, "task_result": load_task_result(f.root, TASK),
        "invocations": [delegate_custody.invocation_record(f.root, row["token"]) for row in f.posts],
    }, indent=2, default=str) + "\n")


def _request(f, surface="multi_model_review"):
    return ReviewRequest(surface=surface, task_id=TASK, goal="Inspect this candidate",
                         retry_key="commit_review:pause-handoff", call_type=surface,
                         session_root=str(f.repo), session_task="Read the candidate and return findings.")


def _slot():
    return ReviewSlot(slot_id="reviewer", model="fake-review=fake-small",
                      route=ReviewRouteKind.AGENT_SESSION,
                      session_target="fake-review=fake-small", timeout_sec=30)


def _run(f, request):
    return run_review_request(request, slots=[_slot()], drive_root=f.root, usage_ctx=f.ctx)


def _start(f, request):
    outcome = {}

    def author():
        try:
            with _parent(f) as parent:
                f.ctx.model_wait_context = parent
                parent.tool_context = f.ctx
                outcome["result"] = _run(f, request)
                from ouroboros.budget_pause import enter_owner_pause
                enter_owner_pause(f.limit)
                outcome["resumed"] = True
        except BaseException as exc:
            outcome["exception"] = repr(exc)

    thread = threading.Thread(target=author, daemon=True)
    f.threads.append(thread)
    thread.start()
    return outcome


def _pause(f, outcome):
    from supervisor.owner_pause_control import request_owner_pause, refresh_owner_pause_tree

    ack = request_owner_pause(TASK, request_id="checkpoint-pause")
    assert ack["ok"], ack
    until(lambda: "result" in outcome or "exception" in outcome)
    assert "exception" not in outcome, outcome
    until(lambda: (load_task_result(f.root, TASK).get("owner_wait") or {}).get("state") == "waiting")
    assert refresh_owner_pause_tree(TASK) == "paused"
    fence = owner_pause.read_fence(f.root, TASK)
    assert TASK in fence["warm_members"] and TASK in fence["finishing_reviews"]
    assert "resumed" not in outcome
    return outcome["result"].actors[0]


def _resume(f, outcome):
    from supervisor.budget_resume import resume_warm_owner_pause_root

    resumed = resume_warm_owner_pause_root(TASK)
    assert resumed["ok"] and resumed["warm"], resumed
    until(lambda: "resumed" in outcome or "exception" in outcome)
    assert "exception" not in outcome, outcome
    grant = owner_pause.read_fence(f.root, TASK)["resume_grant"]
    assert grant["consumed_at"] and grant["consumed_by"] == TASK


def _entry(request):
    return review_custody._ACTIVE[review_custody._attempt_key(request, _slot())]


def _settled(entry):
    assert entry.event.wait(10)
    until(lambda: entry.key not in review_custody._ACTIVE)


@pytest.mark.parametrize("surface", ["multi_model_review", "scope_review"])
@pytest.mark.parametrize("fast_resume", [False, True], ids=["pause-held", "fast-resume"])
def test_fresh_checkpoint_cannot_send_after_author_pause(session, surface, fast_resume):
    f = session
    request = _request(f, surface)
    f.checkpoint_release.clear()
    outcome = _start(f, request)
    assert f.checkpoint_entered.wait(10), outcome
    entry = _entry(request)
    token = entry.retry_state["pending_invocation_id"]
    assert entry.started_at and not owner_pause.episode_armed(entry.episode)
    assert not f.posts and f.checkpoints[0]["invocation_id"] == token
    pending = _pause(f, outcome)
    if fast_resume:
        _resume(f, outcome)
    f.checkpoint_release.set()
    _settled(entry)
    # Assert the transport first: baseline must demonstrate the actual first-POST race.
    assert f.posts == [], "fresh START_REQUESTED bypassed Pause's unsent latch after Resume"
    assert pending["operation_state"] == "pending_dispatch"
    assert pending["usage"]["owner_pause_preparing"] and pending["late_result_pending"]
    assert entry.pause_before_launch and entry.actor.operation_state == "not_dispatched"
    assert not entry.actor.late_result_pending
    assert "pending_invocation_id" not in entry.retry_state
    assert delegate_custody.invocation_record(f.root, token)["state"] == "failed_definite"
    if not fast_resume:
        _resume(f, outcome)
    with _parent(f):
        collected = _run(f, request).actors[0]
    assert collected["operation_id"] == pending["operation_id"]
    assert collected["operation_state"] == "not_dispatched" and not collected["late_result_pending"]
    assert not f.posts, "collection of the refused preparation bought a replacement"


@pytest.mark.parametrize("resume_first", [False, True], ids=["finish-paused", "resume-live"])
def test_started_session_finishes_and_collects_without_another_post(session, resume_first):
    f = session
    request = _request(f)
    f.poll_release.clear()
    outcome = _start(f, request)
    assert f.poll_entered.wait(10), outcome
    entry = _entry(request)
    assert owner_pause.episode_armed(entry.episode) and entry.started_at
    original = copy.deepcopy(f.posts)
    pending = _pause(f, outcome)
    assert pending["operation_state"] == "in_flight" and pending["late_result_pending"]
    assert not entry.pause_before_launch
    if resume_first:
        _resume(f, outcome)
    f.poll_release.set()
    _settled(entry)
    assert entry.actor.status == "ok"
    if not resume_first:
        assert "resumed" not in outcome, "review completion cannot Resume the author"
        _resume(f, outcome)
    with _parent(f):
        collected = _run(f, request).actors[0]
    assert collected["operation_id"] == pending["operation_id"]
    # The exact paid answer the session settled with is what the author collects.
    assert collected["status"] == "ok" and collected["raw_text"] == entry.actor.raw_text
    assert json.loads(collected["raw_text"]) == {"findings": []}
    assert f.posts == original and len(f.posts) == 1
    assert not [cancel for gateway in FakeGateway.instances for cancel in gateway.cancels]


@pytest.mark.parametrize("remote_state", ["started", "unknown"])
def test_recovered_invocation_is_not_reclassified_as_unsent(session, remote_state):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    f = session
    request = _request(f, "advisory_review")  # one attempt; no transport retry in this surface
    if remote_state == "unknown":
        FakeGateway.start_error = ClaudexorUnavailable("daemon_unreachable", "POST outcome unknown", status_code=0)
    else:
        # A terminal-read failure leaves the already-started durable run intact.
        FakeGateway.poll_error = ClaudexorUnavailable("observation_refused", "unreadable run", status_code=403)
    with _parent(f):
        first = _run(f, request).actors[0]
    key = review_custody._attempt_key(request, _slot())
    until(lambda: key not in review_custody._ACTIVE)
    assert first["operation_state"] == "in_flight" and first["late_result_pending"]
    retry = f.ctx._review_pending_invocations[key]
    token, operation_id = retry["pending_invocation_id"], retry["operation_id"]
    original = copy.deepcopy(delegate_custody.invocation_record(f.root, token))
    assert original["state"] == ("pending" if remote_state == "unknown" else "started")
    assert len(f.posts) == 1
    f.checkpoint_entered.clear()
    f.poll_entered.clear()
    f.checkpoint_release.clear()
    f.poll_release.clear()
    outcome = _start(f, request)
    boundary = f.checkpoint_entered if remote_state == "unknown" else f.poll_entered
    assert boundary.wait(10), outcome
    entry = _entry(request)
    assert entry.started_at == "" and not owner_pause.episode_armed(entry.episode)
    pending = _pause(f, outcome)
    assert pending["operation_state"] == "in_flight" and pending["late_result_pending"]
    assert not entry.pause_before_launch and not entry.episode.get("pause_before_launch")
    assert pending["usage"]["pending_invocation_id"] == token
    f.checkpoint_release.set()
    f.poll_release.set()
    _settled(entry)
    assert len(f.posts) == 1, "recovery under Pause sent another POST"
    if remote_state == "unknown":
        assert entry.actor.operation_state == "custody_lost" and entry.actor.late_result_pending
        assert entry.retry_state["pending_invocation_id"] == token
        retained = delegate_custody.invocation_record(f.root, token)
        assert retained["state"] == "pending" and retained["request"] == original["request"]
        assert not [project for gateway in FakeGateway.instances for project in gateway.removals]
    else:
        assert entry.actor.status == "ok" and entry.actor.operation_id == operation_id
    _resume(f, outcome)
    with _parent(f):
        collected = _run(f, request).actors[0]
    assert collected["operation_id"] == operation_id and len(f.posts) == 1
    assert collected["operation_state"] == ("custody_lost" if remote_state == "unknown" else "late_settled")
