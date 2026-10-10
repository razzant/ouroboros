"""Owner 2026-10-08 (full variant): the author is truly paused; its launched reviewer finishes.

Real consumers: the review substrate runs the panel, each reviewer round is a real
physical attempt through the ledger and ``submit_model``, the author's synchronous
drain detaches, the census reads the durable fence and the Resume consumes its
grant. A reviewer that had not launched when the Pause landed never starts; one
that launched may finish its episode (further rounds included) but nothing new
starts and the author commits nothing.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

from ouroboros import owner_pause, review_pause
from ouroboros import usage_accounting as ua
from ouroboros.task_results import write_task_result
from tests.test_review_operation_lifetime import TASK, _parent, _request, _slot, env, until  # noqa: F401
from tests._usage_store_testing import attempt_rows_in_start_order

pytestmark = pytest.mark.serial


def _extract(_response):
    return {"prompt_tokens": 3, "completion_tokens": 2}, .01, True


class EpisodeModel:
    """A reviewer whose one check takes two physical rounds (an inspection follow-up)."""

    def __init__(self, *, gate_before_first=None):
        self.first_sent, self.release_first = threading.Event(), threading.Event()
        self.gate_before_first = gate_before_first
        self.preparing = threading.Event()
        self.rounds = []

    def _round(self, label, wait=None):
        def send():
            self.rounds.append((label, time.monotonic()))
            if wait is not None:
                self.first_sent.set()
                assert wait.wait(10)
            return {"round": label}
        return ua.execute_physical_attempt(ua.AttemptRequest(model="m", provider="test", reservation_usd=.05),
                                           send, extractor=_extract)

    def chat(self, messages, model, model_role="", **kwargs):
        if self.gate_before_first is not None:
            self.preparing.set()
            assert self.gate_before_first.wait(10)
        self._round("first", wait=self.release_first)
        self._round("follow-up")
        return ({"content": json.dumps({"verdict": "PASS", "findings": [], "summary": "checked"})},
                {"prompt_tokens": 3, "completion_tokens": 2})


@pytest.fixture
def paused_env(env, monkeypatch):  # noqa: F811 - the imported fixture
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    monkeypatch.setattr(review_pause, "DETACH_POLL_SEC", 0.05)
    return env


def _author(f, model, request, outcome):
    def run():
        with _parent(f):
            outcome["result"] = review_operation_run(f, model, request)
            outcome["returned_at"] = time.monotonic()
    thread = threading.Thread(target=run)
    thread.start()
    return thread


def review_operation_run(f, model, request):
    from ouroboros.review_substrate import run_review_request

    return run_review_request(request, slots=[_slot()], drive_root=f.root, usage_ctx=f.ctx, llm=model)


def test_a_launched_reviewer_finishes_its_episode_while_the_synchronous_author_detaches(paused_env):
    f, model, outcome = paused_env, EpisodeModel(), {}
    thread = _author(f, model, _request(drain=False), outcome)
    try:
        assert model.first_sent.wait(10)
        paused_at = time.monotonic()
        owner_pause.install_fence(f.root, TASK, request_id="pause")
        until(lambda: "returned_at" in outcome)
        actor = outcome["result"].actors[0]
        # The author took a PENDING row with it: no verdict, nothing committed.
        assert actor["operation_state"] == "in_flight" and actor["late_result_pending"]
        assert actor["usage"].get("owner_pause_detached") is True
        assert not (actor.get("parsed") or {}).get("verdict")
        detached = review_pause.live_detached_operations(TASK)
        assert len(detached) == 1, "the operation stays live in this process"
        owner_id = detached[0]["owner_id"]
        assert review_pause.review_operation_state(owner_id) == "running"
        # The author's own context is still fenced: it cannot start anything.
        from ouroboros.llm_attempt import PhysicalDispatchInterrupted, require_physical_dispatch_window
        with ua.usage_scope(ua.UsageScope(drive_root=f.root, task_id=TASK, root_task_id=TASK)):
            with pytest.raises(PhysicalDispatchInterrupted):
                require_physical_dispatch_window()
    finally:
        model.release_first.set()
        thread.join(10)
    until(lambda: review_pause.review_operation_state(owner_id) == "closed")
    rounds = [label for label, _at in model.rounds]
    assert rounds == ["first", "follow-up"], "the started episode finished its own rounds"
    assert model.rounds[1][1] > paused_at, "its follow-up round ran after the Pause"
    assert all(row["state"] == "settled" for row in attempt_rows_in_start_order(f.root))
    assert review_pause.live_detached_operations(TASK) == []
    # The same unchanged request after Resume collects it without a new dispatch.
    owner_pause.release_fence(f.root, TASK, reason="owner_resume")
    with _parent(f):
        again = review_operation_run(f, model, _request(drain=False))
    assert (again.actors[0].get("parsed") or {}).get("verdict") == "PASS", again.actors[0]
    assert [label for label, _ in model.rounds] == ["first", "follow-up"], "no second purchase"


@pytest.mark.parametrize("preparation_boundary", ["model_preparation", "before_episode_binding"])
def test_a_reviewer_that_had_not_launched_when_the_pause_landed_never_starts(
        paused_env, monkeypatch, preparation_boundary):
    from contextlib import contextmanager

    gate = threading.Event()
    f, outcome = paused_env, {}
    model = EpisodeModel(gate_before_first=gate if preparation_boundary == "model_preparation" else None)
    if preparation_boundary == "before_episode_binding":
        slot_episode = review_pause.slot_episode

        @contextmanager
        def delayed_episode(entry, scope):
            model.preparing.set()
            assert gate.wait(10)
            with slot_episode(entry, scope) as marker:
                yield marker

        monkeypatch.setattr(review_pause, "slot_episode", delayed_episode)
    thread = _author(f, model, _request(drain=False), outcome)
    try:
        assert model.preparing.wait(10)
        owner_pause.install_fence(f.root, TASK, request_id="pause")
        until(lambda: "returned_at" in outcome, timeout=1.5)
        actor = outcome["result"].actors[0]
        assert actor["operation_state"] == "pending_dispatch" and actor["late_result_pending"]
        assert actor["usage"]["owner_pause_preparing"] is True
        assert "preparation" in actor["error"].lower() and "already running" not in actor["error"]
        assert not model.rounds and review_pause.live_detached_operations(TASK)
        from ouroboros.review_custody import _ACTIVE
        entry = next(entry for entry in _ACTIVE.values() if entry.operation_id == actor["operation_id"])
        if preparation_boundary == "before_episode_binding":
            owner_pause.release_fence(f.root, TASK, reason="owner_resume")
    finally:
        gate.set()
        model.release_first.set()
        thread.join(10)
    until(lambda: not review_pause.live_detached_operations(TASK))
    assert entry.actor.operation_state == "not_dispatched", vars(entry.actor)
    owner_pause.release_fence(f.root, TASK, reason="owner_resume")
    with _parent(f):
        actor = review_operation_run(f, model, _request(drain=False)).actors[0]
    assert not model.rounds, "no byte was sent after the Pause"
    assert actor["status"] != "ok" and not actor.get("late_result_pending")
    assert actor["operation_state"] == "not_dispatched"
    assert all(row["state"] == "released" for row in attempt_rows_in_start_order(f.root))


def test_pause_detaches_a_mixed_panel_and_resume_cannot_launch_its_unsent_preparation(paused_env):
    from dataclasses import replace
    from ouroboros.review_substrate import run_review_request

    f, outcome, gate = paused_env, {}, threading.Event()
    armed, preparing = EpisodeModel(), EpisodeModel(gate_before_first=gate)
    slots = [replace(_slot("armed"), model="armed"), replace(_slot("preparing"), model="preparing")]

    class Panel:
        def chat(self, messages, model, **kwargs):
            return {"armed": armed, "preparing": preparing}[model].chat(messages, model, **kwargs)

    panel = Panel()

    def run():
        with _parent(f):
            outcome["result"] = run_review_request(_request(drain=False), slots=slots, drive_root=f.root,
                                                    usage_ctx=f.ctx, llm=panel)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert armed.first_sent.wait(10) and preparing.preparing.wait(10)
        owner_pause.install_fence(f.root, TASK, request_id="mixed")
        until(lambda: "result" in outcome, timeout=1.5)
        rows = {row["slot_id"]: row for row in outcome["result"].actors}
        assert rows["armed"]["operation_state"] == "in_flight"
        assert rows["preparing"]["operation_state"] == "pending_dispatch"
        assert rows["preparing"]["usage"]["owner_pause_preparing"]
        assert review_pause.live_detached_operations(TASK), "both workers still belong to the warm operation"
        owner_pause.release_fence(f.root, TASK, reason="owner_resume")
        assert not preparing.rounds, "Resume did not dispatch an unstarted slot"
    finally:
        gate.set()
        preparing.release_first.set()
        armed.release_first.set()
        thread.join(10)
    until(lambda: not review_pause.live_detached_operations(TASK))
    assert [label for label, _ in armed.rounds] == ["first", "follow-up"]
    assert preparing.rounds == [], "the preparation refused at Pause cannot send after a fast Resume"
    with _parent(f):
        replay = run_review_request(_request(drain=False), slots=slots, drive_root=f.root,
                                    usage_ctx=f.ctx, llm=panel)
    rows = {row["slot_id"]: row for row in replay.actors}
    assert rows["armed"]["parsed"]["verdict"] == "PASS"
    assert rows["preparing"]["operation_state"] == "not_dispatched"
    assert not rows["preparing"]["late_result_pending"]
    assert [label for label, _ in armed.rounds] == ["first", "follow-up"] and not preparing.rounds


def test_a_new_panel_under_an_accepted_pause_is_refused_before_any_send(paused_env):
    f, model = paused_env, EpisodeModel()
    owner_pause.install_fence(f.root, TASK, request_id="pause")
    with _parent(f):
        result = review_operation_run(f, model, _request(drain=False))
    actor = result.actors[0]
    assert actor["operation_state"] == "not_dispatched" and "NOT STARTED" in str(actor.get("error"))
    assert not model.rounds and attempt_rows_in_start_order(f.root) == []


@pytest.mark.parametrize("capture_state", ["settled", "unresolved"])
def test_preparation_refusal_preserves_independent_positive_or_unknown_capture(paused_env, capture_state):
    from types import SimpleNamespace
    from ouroboros.review_custody import _ACTIVE

    preparing, release, outcome = threading.Event(), threading.Event(), {}

    class RetainedCaptureModel:
        def chat(self, *_a, **_kw):
            preparing.set()
            assert release.wait(10)
            error = RuntimeError("independent provider evidence retained by the adapter")
            error.physical_attempt_capture = SimpleNamespace(
                state=capture_state, attempt_id="prior-exact-attempt", provider_status_code=None)
            raise error

    f = paused_env
    thread = _author(f, RetainedCaptureModel(),
                     _request(retry_key=f"pause-capture:{capture_state}", drain=False), outcome)
    try:
        assert preparing.wait(10)
        owner_pause.install_fence(f.root, TASK, request_id="before-capture-delivery")
        until(lambda: "result" in outcome, timeout=1.5)
        row = outcome["result"].actors[0]
        entry = next(entry for entry in _ACTIVE.values() if entry.operation_id == row["operation_id"])
        assert entry.pause_before_launch
    finally:
        release.set()
        thread.join(10)
    until(lambda: not review_pause.live_detached_operations(TASK))
    assert entry.actor.usage["physical_attempt_state"] == capture_state
    assert entry.actor.operation_state == ("custody_lost" if capture_state == "unresolved" else "late_settled")
    assert entry.actor.late_result_pending is (capture_state == "unresolved")


def test_the_tree_reads_paused_with_its_reviewer_finishing_and_resume_consumes_its_grant_once(
        paused_env, monkeypatch):
    from supervisor.budget_resume import resume_warm_owner_pause_root
    from supervisor.owner_pause_control import refresh_owner_pause_tree, request_owner_pause
    from tests._budget_pause_exact_helpers import _install_queue
    from ouroboros.owner_wait import consume_warm_resume

    f, model, outcome = paused_env, EpisodeModel(), {}
    queue, state, workers = _install_queue(f.root, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    workers.RUNNING[TASK] = {"task": {"id": TASK, "type": "task", "chat_id": 7, "root_task_id": TASK},
                             "worker_id": 0, "attempt": 1}
    thread = _author(f, model, _request(drain=False), outcome)
    try:
        assert model.first_sent.wait(10)
        ack = request_owner_pause(TASK, request_id="press")
        assert ack["ok"] and ack["state"] == "requested", "the author still runs: Pausing"
        until(lambda: "returned_at" in outcome)
        # The author parks warm (its worker keeps the stack) while its reviewer runs.
        workers.RUNNING[TASK]["owner_wait"] = {"state": "waiting", "reason": "owner_pause", "wait_id": "w1",
                                               "owner_pause": {"fence_id": ack["fence_id"]}}
        assert refresh_owner_pause_tree(TASK) == "paused"
        fence = owner_pause.read_fence(f.root, TASK)
        assert fence["warm_members"] == [TASK] and TASK in fence["finishing_reviews"]
        # Resume does not wait for the reviewer.
        resumed = resume_warm_owner_pause_root(TASK)
        assert resumed["ok"] and resumed["warm"], resumed
        ctx = type("Ctx", (), {"task_id": TASK, "task_attempt": 1, "budget_drive_root": f.root,
                               "drive_root": f.root, "task_metadata": {}, "root_task_id": TASK})()
        assert consume_warm_resume(ctx) is True
        grant = owner_pause.read_fence(f.root, TASK)["resume_grant"]
        assert grant["consumed_by"] == TASK and grant["consumed_at"]
        stamp = grant["consumed_at"]
        assert consume_warm_resume(ctx) is True
        assert owner_pause.read_fence(f.root, TASK)["resume_grant"]["consumed_at"] == stamp, "spent once"
        assert review_pause.live_detached_operations(TASK), "the reviewer still runs after Resume"
    finally:
        model.release_first.set()
        thread.join(10)


def test_the_owner_sees_the_reviews_its_pause_lets_finish_once_each_way(tmp_path, monkeypatch):
    from supervisor import owner_pause_control

    write_task_result(tmp_path, TASK, "running", chat_id=7, root_task_id=TASK)
    sent = []
    monkeypatch.setattr("supervisor.message_bus.send_with_budget",
                        lambda chat_id, text, **kw: sent.append((chat_id, text, kw)))
    owner_pause_control._announce_finishing_reviews(tmp_path, TASK, [], ["review-a"])
    owner_pause_control._announce_finishing_reviews(tmp_path, TASK, ["review-a"], ["review-a"])  # unchanged
    owner_pause_control._announce_finishing_reviews(tmp_path, TASK, ["review-a"], [])
    assert [chat for chat, _text, _kw in sent] == [7, 7]
    assert "Review work is finishing separately; Resume does not wait" in sent[0][1]
    assert "Resume does not wait" in sent[0][1] and "Nothing is committed" in sent[0][1]
    assert "has ended" in sent[1][1] and "stays paused until Resume" in sent[1][1]
    assert all(kw["is_progress"] and kw["system_type"] == "owner_pause_notice" for _c, _t, kw in sent)


def test_a_new_pause_before_the_warm_stack_woke_keeps_it_parked(paused_env, monkeypatch):
    from ouroboros.owner_wait import consume_warm_resume

    f = paused_env
    first, _ = owner_pause.install_fence(f.root, TASK, request_id="p1")
    owner_pause.release_fence(f.root, TASK, reason="owner_resume")
    owner_pause.install_fence(f.root, TASK, request_id="p2")  # paused again before the wake
    ctx = type("Ctx", (), {"task_id": TASK, "task_attempt": 1, "budget_drive_root": f.root,
                           "drive_root": f.root, "task_metadata": {}, "root_task_id": TASK})()
    assert consume_warm_resume(ctx) is False, "the stack parks again under the newer Pause"


def test_a_start_request_row_is_not_a_send_the_post_still_meets_the_fence(tmp_path, monkeypatch):
    """Barrier: the durable ``START_REQUESTED`` row precedes the delegated POST, which
    is gated again at its final start point; only that submission arms the episode."""
    from types import SimpleNamespace

    from ouroboros.owner_pause import OwnerPauseRefused, admit_delegated_start, episode_armed, run_operation

    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    source = SimpleNamespace(drive_root=tmp_path, task_id="root", root_task_id="root")
    posted = []
    with owner_pause.review_episode("root") as marker:
        assert admit_delegated_start(tmp_path, {"task_id": "root", "root_task_id": "root", "invocation_id": "i1"})
        assert not episode_armed(marker), "a start request is custody, not the send"
        owner_pause.install_fence(tmp_path, "root", request_id="pause-before-post")
        with pytest.raises(OwnerPauseRefused):
            run_operation(source, lambda: posted.append("post"))
    assert posted == [], "a reviewer that had not sent never starts under the Pause"

    owner_pause.release_fence(tmp_path, "root", reason="owner_resume")
    with owner_pause.review_episode("root") as marker:
        run_operation(source, lambda: posted.append("post"))
        assert episode_armed(marker)
        owner_pause.install_fence(tmp_path, "root", request_id="pause-after-post")
        run_operation(source, lambda: posted.append("follow-up"))  # the same started episode
    with pytest.raises(OwnerPauseRefused):  # outside the episode: the author stays fenced
        run_operation(source, lambda: posted.append("author"))
    assert posted == ["post", "follow-up"]


def test_the_model_handoff_is_the_last_fence_gate_and_arms_before_any_byte_leaves(tmp_path, monkeypatch):
    """Barrier for the parent's question: ``submit_model`` arms the episode at the
    exact sender's executor handoff — under the launch lock the Pause's own fence
    write takes — not at a socket write. A Pause before the handoff refuses it at
    $0 (nothing queued, unarmed). After the handoff a QUEUED sender that has not
    run yet still owns its bytes (``model_handed_off``): no later gate can withdraw
    it, which is exactly why its episode counts as started. The author stays fenced."""
    from types import SimpleNamespace

    from ouroboros.llm_attempt import _PhysicalSendNotStarted
    from ouroboros.owner_pause import OwnerPauseRefused, episode_armed, run_operation

    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    source = SimpleNamespace(drive_root=tmp_path, task_id="root", root_task_id="root")
    queued, ran = [], []

    def submit(run, function):  # an executor whose worker has not picked the future up yet
        queued.append((run, function))
        return "queued-future"

    def sender():
        ran.append((owner_pause.model_handed_off("a2"), owner_pause.read_fence(tmp_path, "root")["state"]))

    owner_pause.install_fence(tmp_path, "root", request_id="before-handoff")
    with owner_pause.review_episode("root") as marker:
        with pytest.raises(_PhysicalSendNotStarted):
            owner_pause.submit_model(SimpleNamespace(scope=source, attempt_id="a1"), submit, sender)
    assert not queued and not episode_armed(marker), "refused before the handoff: never queued, never armed"

    owner_pause.release_fence(tmp_path, "root", reason="owner_resume")
    with owner_pause.review_episode("root") as marker:
        assert owner_pause.submit_model(SimpleNamespace(scope=source, attempt_id="a2"), submit, sender) == "queued-future"
        assert episode_armed(marker) and not ran, "armed at the handoff, before the sender ran"
        owner_pause.install_fence(tmp_path, "root", request_id="after-handoff")
        run, function = queued[-1]
    run(function)
    assert ran == [(True, "requested")], "the handed sender proceeds although the Pause is closed"
    with pytest.raises(OwnerPauseRefused):
        run_operation(source, lambda: None)  # the author, outside the episode


@pytest.fixture
def direct_commit(paused_env, monkeypatch):
    """Real commit handler/stage/parallel review; preparation and Git effects are inert.

    The model still crosses the physical-attempt ledger and final sender handoff.
    No subprocess, commit, tag, push, provider or production state is touched.
    """
    from types import SimpleNamespace

    from ouroboros import config
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.tools import git, parallel_review, review
    from ouroboros.tools.scope_review_contract import SCOPE_REQUIRED_ITEMS
    from ouroboros.triad_review import REVIEW_TWO_PART_OBJECT_CONTRACT

    f = paused_env
    f.ctx.repo_dir = f.root / "candidate"
    f.ctx.repo_dir.mkdir()
    f.ctx.current_task_type = "task"
    f.ctx.emit_progress_fn = lambda *_a, **_kw: None
    effects, records, locks = [], [], []

    def command(argv, **_kw):
        effects.append(tuple(argv))
        return "candidate-head" if argv[1:3] == ["rev-parse", "HEAD"] else ""

    monkeypatch.setattr(git, "run_cmd", command)
    monkeypatch.setattr(parallel_review, "run_cmd", command)
    monkeypatch.setattr("supervisor.update_merge.managed_assisted_tx_for", lambda *_a: (None, ""))
    monkeypatch.setattr(git, "_task_attributed_commit_paths", lambda *_a: (["candidate.py"], None, "", None))
    monkeypatch.setattr(git, "_acquire_git_lock", lambda *_a: "lock")
    monkeypatch.setattr(git, "_release_git_lock", lambda token: locks.append(token))
    monkeypatch.setattr(git, "_prepare_review_commit_worktree", lambda *_a: (False, ""))
    monkeypatch.setattr(git, "_stage_candidate_for_review", lambda *_a, **_kw: (["candidate.py"], ["candidate.py"], None))
    monkeypatch.setattr(git, "protected_paths_in", lambda *_a: [])
    fingerprint = {"ok": True, "fingerprint": "unchanged-candidate", "binding": {"tree_sha": "tree"}}
    monkeypatch.setattr(git, "_fingerprint_staged_diff", lambda *_a: fingerprint)
    monkeypatch.setattr(git, "commit_review_contract_fingerprint", lambda: "test-commit-contract")
    monkeypatch.setattr(git, "_preflight_and_tests_gate", lambda *_a, **_kw: None)
    monkeypatch.setattr("ouroboros.review_state.compute_snapshot_hash", lambda *_a, **_kw: "unchanged-candidate")
    monkeypatch.setattr("ouroboros.review_evidence.capture_commit_review_evidence", lambda *_a: {})
    record = git._record_commit_attempt

    def record_attempt(*a, **kw):
        records.append({"status": a[2], **kw})
        return record(*a, **kw)

    monkeypatch.setattr(git, "_record_commit_attempt", record_attempt)
    monkeypatch.setattr(git, "_verify_reviewed_commit_binding", lambda *_a, **_kw: (True, ""))
    monkeypatch.setattr(git, "_auto_tag_on_version_bump", lambda *_a, **_kw: effects.append(("tag",)) or "")
    monkeypatch.setattr(git, "_managed_post_commit_tests_gate", lambda *_a, **_kw: None)
    monkeypatch.setattr(git, "_post_commit_result", lambda *_a, **_kw: effects.append(("post-commit-tests",)))
    monkeypatch.setattr(git, "_auto_push", lambda *_a: effects.append(("push",)) or "")
    monkeypatch.setattr(git, "_publish_reviewed_commit", lambda _ctx, _msg, _sha, tag, _warning, _paths, push: "committed" + tag + push)
    monkeypatch.setattr(git, "record_bound_commit_success", lambda *_a: None)
    monkeypatch.setattr("ouroboros.body_candidate.record_reviewed_commit", lambda *_a: None)
    monkeypatch.setattr(config, "get_review_enforcement", lambda: "blocking")
    monkeypatch.setattr("ouroboros.tools.review_admission.admit_commit_gate_wave", lambda *_a: None)
    # The one commit wave: a retrieving seat asked both parts of the one brief (contract B).
    plan = {"models": ["m"], "routes": [ReviewRouteKind.API_CHAT], "efforts": ["high"],
            "slot_ids": ["commit-reviewer"], "use_local": [False], "subagent_ids": [""],
            "session_targets": [""], "session_profiles": [""], "retrieves": [True],
            "parts": [("change", "coupling")], "session_tasks": ["Review this candidate."],
            "session_policies": [{"output_contract": REVIEW_TWO_PART_OBJECT_CONTRACT}], "brief_shas": ["brief-sha"]}
    prepared = {"blocking_review": True, "prompt": "Review this candidate.", "models": ["m"],
                "stable_prefix_len": 0, "routes": plan["routes"], "session_task": "", "target_repo": f.ctx.repo_dir,
                "row_plan": plan, "layer": "core", "brief_texts": {"brief-sha": "Review this candidate."}}
    monkeypatch.setattr(review, "_prepare_unified_review", lambda *_a, **_kw: (dict(prepared), None, False))
    answer = json.dumps({"change": [], "change_clean": True, "coupling": [
        {"item": item, "verdict": "PASS", "severity": "advisory",
         "reason": "checked the relevant code path and its consumers thoroughly"}
        for item in sorted(SCOPE_REQUIRED_ITEMS)]})

    class CommitModel(EpisodeModel):
        def chat(self, *a, **kw):
            super().chat(*a, **kw)
            return {"content": answer}, {"prompt_tokens": 3, "completion_tokens": 2}

    model = CommitModel()
    monkeypatch.setattr(review, "LLMClient", lambda: model)
    return SimpleNamespace(f=f, model=model, git=git, effects=effects, records=records, locks=locks)


def test_direct_commit_returns_pending_on_pause_and_rejoins_the_same_review(direct_commit):
    d, outcome = direct_commit, {}
    resumed, resumed_thread = {}, None

    def commit():
        with _parent(d.f):
            outcome["result"] = d.git._commit_reviewed(d.f.ctx, "candidate")

    thread = threading.Thread(target=commit)
    thread.start()
    try:
        assert d.model.first_sent.wait(10)
        owner_pause.install_fence(d.f.root, TASK, request_id="during-commit-review")
        until(lambda: "result" in outcome)
        assert "REVIEW_PENDING" in outcome["result"], outcome
        assert not any(row[0] in {"tag", "push"} or row[1:2] == ("commit",) for row in d.effects)
        assert d.locks == ["lock"], "the author released its Git lock while the reviewer remains running"
        pending = d.f.ctx._last_triad_raw_results[0]
        assert pending["operation_state"] == "in_flight" and pending["late_result_pending"]
        assert review_pause.live_detached_operations(TASK)
        assert d.records[-1]["phase"] == "late_wait"
        # The owner can Resume before the reviewer ends; collection will wait on
        # the same operation, without another physical send.
        owner_pause.release_fence(d.f.root, TASK, reason="owner_resume")
        assert review_pause.live_detached_operations(TASK)

        def rejoin():
            with _parent(d.f):
                resumed["result"] = d.git._commit_reviewed(d.f.ctx, "candidate")

        resumed_thread = threading.Thread(target=rejoin)
        resumed_thread.start()
        until(lambda: getattr(d.f.ctx, "_review_reconcile_only", False))
        assert "result" not in resumed and len(d.model.rounds) == 1, "Resume rejoined the already-running attempt"
    finally:
        d.model.release_first.set()
        thread.join(10)
        if resumed_thread is not None:
            resumed_thread.join(10)
    until(lambda: not review_pause.live_detached_operations(TASK))
    result = resumed["result"]
    assert result.startswith("committed"), result
    assert d.f.ctx._last_triad_raw_results[0]["operation_id"] == pending["operation_id"]
    assert [label for label, _ in d.model.rounds] == ["first", "follow-up"], "unchanged review collected at $0"
    assert sum(row[1:2] == ("commit",) for row in d.effects) == 1


def test_direct_commit_rechecks_pause_after_review_before_its_final_git_effect(direct_commit, monkeypatch):
    d = direct_commit
    d.model.release_first.set()
    stage = d.git._run_reviewed_stage_cycle

    def pause_after_review(*a, **kw):
        outcome = stage(*a, **kw)
        assert outcome["status"] == "passed", outcome
        owner_pause.install_fence(d.f.root, TASK, request_id="after-review-before-commit")
        return outcome

    monkeypatch.setattr(d.git, "_run_reviewed_stage_cycle", pause_after_review)
    with _parent(d.f):
        result = d.git._commit_reviewed(d.f.ctx, "candidate")
    assert "OWNER_PAUSE" in result, result
    assert not any(row[0] in {"tag", "push"} or row[1:2] == ("commit",) for row in d.effects), d.effects
    assert d.locks == ["lock"]


def test_pause_before_the_panel_launches_refuses_it_at_zero_and_never_passes(direct_commit, monkeypatch):
    """Pause after the author's preflight, before the panel's first send: the new panel
    never starts, its reserved row reads a $0 refusal (not lost custody), and an
    unreviewed candidate is never treated as reviewed. After Resume the same commit runs."""
    d = direct_commit
    d.model.release_first.set()
    gate = d.git._preflight_and_tests_gate

    def paused_after_preflight(*a, **kw):
        owner_pause.install_fence(d.f.root, TASK, request_id="after-preflight")
        return gate(*a, **kw)

    monkeypatch.setattr(d.git, "_preflight_and_tests_gate", paused_after_preflight)
    with _parent(d.f):
        result = d.git._commit_reviewed(d.f.ctx, "candidate")
    assert result.startswith("⚠️ REVIEW_BLOCKED") and "OWNER_PAUSE_NOT_STARTED" in result, result
    assert "CUSTODY_LOST" not in result and d.f.ctx._last_review_verdict["aggregate"] == "NOT_DISPATCHED"
    assert d.model.rounds == [], "no reviewer was sent"
    row = d.f.ctx._last_triad_raw_results[0]
    assert row["operation_state"] == "not_dispatched" and not row["late_result_pending"]
    assert d.records[-1]["status"] == "blocked" and d.records[-1]["block_reason"] == "review_quorum"
    assert not any(row[0] in {"tag", "push"} or row[1:2] == ("commit",) for row in d.effects), d.effects
    assert all(row["state"] == "settled" for row in attempt_rows_in_start_order(d.f.root))
    owner_pause.release_fence(d.f.root, TASK, reason="owner_resume")
    monkeypatch.setattr(d.git, "_preflight_and_tests_gate", gate)
    with _parent(d.f):
        result = d.git._commit_reviewed(d.f.ctx, "candidate")
    assert result.startswith("committed"), result
    assert [label for label, _ in d.model.rounds] == ["first", "follow-up"]
    assert sum(row[1:2] == ("commit",) for row in d.effects) == 1


@pytest.mark.parametrize("pause_boundary", ["commit", "tag"])
def test_pause_after_commit_handoff_preserves_that_commit_and_withholds_later_publication(
        direct_commit, monkeypatch, pause_boundary):
    d, outcome = direct_commit, {}
    d.model.release_first.set()
    entered, finish = threading.Event(), threading.Event()
    command = d.git.run_cmd

    def held_commit(argv, **kw):
        if argv[1:2] == ["commit"] and pause_boundary == "commit":
            entered.set()
            assert finish.wait(10)
        return command(argv, **kw)

    monkeypatch.setattr(d.git, "run_cmd", held_commit)
    if pause_boundary == "tag":
        tag = d.git._auto_tag_on_version_bump

        def held_tag(*a, **kw):
            entered.set()
            assert finish.wait(10)
            return tag(*a, **kw)

        monkeypatch.setattr(d.git, "_auto_tag_on_version_bump", held_tag)

    def commit():
        with _parent(d.f):
            outcome["result"] = d.git._commit_reviewed(d.f.ctx, "candidate")

    thread = threading.Thread(target=commit)
    thread.start()
    try:
        assert entered.wait(10)
        owner_pause.install_fence(d.f.root, TASK, request_id="after-commit-handoff")
        assert "result" not in outcome, "the admitted command still owns its result"
    finally:
        finish.set()
        thread.join(10)
    assert outcome["result"].startswith("committed"), outcome
    assert "owner pause" in outcome["result"].lower(), outcome
    assert d.f.ctx.last_reviewed_commit_sha == "candidate-head"
    assert sum(row[1:2] == ("commit",) for row in d.effects) == 1
    assert not any(row[0] == "push" for row in d.effects), d.effects
    assert not any(row[0] == "post-commit-tests" for row in d.effects), d.effects
    assert sum(row[0] == "tag" for row in d.effects) == (pause_boundary == "tag")
