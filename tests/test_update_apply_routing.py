"""Compact routing and writer-fence tests for managed updates."""

import asyncio
import inspect
import json
import threading
from types import SimpleNamespace

import pytest
from starlette.responses import JSONResponse

from ouroboros.gateway import control


BASE = "a" * 40
TARGET = "b" * 40


@pytest.fixture(autouse=True)
def _reset_writer_admission():
    import supervisor.workers as workers

    workers.open_repo_writer_admission()
    yield
    workers.open_repo_writer_admission()


def _plan(**over):
    plan = {
        "available": True,
        "kind": "clean",
        "local_dirty_count": 0,
        "base_sha": BASE,
        "target_sha": TARGET,
        "target_ref": "managed/main",
        "update_channel": "stable",
        "local_snapshot": BASE,
        "merge_commit": TARGET,
        "code_conflict_paths": [],
        "doc_conflict_paths": [],
        "hot_code_paths": [],
    }
    plan.update(over)
    return plan


class _Request:
    def __init__(self, body=None):
        self._body = body or {}
        self.app = SimpleNamespace(state=SimpleNamespace(request_restart=lambda: None))

    async def json(self):
        return self._body


def _body(response):
    return json.loads(response.body)


def test_clean_plan_requires_real_zero_and_no_conflicts():
    assert control._plan_is_clean(_plan()) is True
    assert control._plan_is_clean(_plan(local_dirty_count="0")) is False
    assert control._plan_is_clean(_plan(local_dirty_count=None)) is False
    assert control._plan_is_clean(_plan(code_conflict_paths=["BIBLE.md"])) is False
    assert control._plan_is_clean(_plan(merge_commit="c" * 40)) is True
    assert control._plan_is_clean(_plan(merge_commit="")) is False


def test_update_has_no_filename_based_protected_route():
    assert not hasattr(control, "_managed_update_protected_block")
    assert not hasattr(control, "_official_protected_hits")


def test_other_owner_repo_mutations_share_the_managed_update_lock():
    for endpoint in (control.api_reset, control._git_rollback_fenced, control.api_git_promote):
        assert "_acquire_repo_mutation_lock" in inspect.getsource(endpoint)
    # Reset must not unlink the directory containing the lock while holding it.
    reset_source = inspect.getsource(control.api_reset)
    assert '"locks"' not in reset_source


def test_manual_restore_uses_writer_fence_before_reset(monkeypatch):
    import supervisor.git_ops as git_ops

    lock = object()
    released = []
    monkeypatch.setattr(control, "_acquire_repo_mutation_lock", lambda: (lock, None))
    monkeypatch.setattr(control, "_release_repo_mutation_lock", released.append)
    monkeypatch.setattr(
        git_ops, "git_capture", lambda _cmd: (0, TARGET, "")
    )
    monkeypatch.setattr(
        control, "_quiesce_repo_writers", lambda _reason: ["active:direct_chat"]
    )
    monkeypatch.setattr(
        git_ops,
        "rollback_to_version",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("reset must not run when writer drain is blocked")
        ),
    )

    response = asyncio.run(control.api_git_rollback(_Request({"target": "v1"})))

    assert response.status_code == 409
    assert _body(response)["reason"] == "update_writer_fence_blocked"
    assert released == [lock]


def test_repo_mutation_lock_contention_is_a_clear_409(monkeypatch):
    import supervisor.update_merge as update_merge

    monkeypatch.setattr(
        update_merge,
        "acquire_update_lock",
        lambda: (_ for _ in ()).throw(RuntimeError("held")),
    )

    lock_fh, error = control._acquire_repo_mutation_lock()

    assert lock_fh is None
    assert error.status_code == 409
    assert "already changing" in _body(error)["error"]


def test_repo_mutations_stay_blocked_for_the_full_durable_update(monkeypatch):
    import supervisor.update_merge as update_merge

    lock = object()
    released = []
    monkeypatch.setattr(update_merge, "acquire_update_lock", lambda: lock)
    monkeypatch.setattr(update_merge, "active_update_tx", lambda: {"phase": "assisted_resolution"})
    monkeypatch.setattr(update_merge, "release_update_lock", lambda value: released.append(value))

    lock_fh, error = control._acquire_repo_mutation_lock()

    assert lock_fh is None
    assert error.status_code == 409
    assert "transaction is still active" in _body(error)["error"]
    assert released == [lock]


def test_preflight_returns_only_merge_plan(monkeypatch):
    import supervisor.update_merge as update_merge

    monkeypatch.setattr(update_merge, "plan_managed_update_merge", lambda **_kwargs: _plan())

    payload = _body(asyncio.run(control.api_update_preflight(None)))

    assert payload == {"merge_plan": _plan()}


def test_fetching_update_endpoints_leave_the_event_loop(monkeypatch):
    calls = []

    async def fake_to_thread(fn, *args, **kwargs):
        calls.append((fn, args, kwargs))
        return {"available": False}

    monkeypatch.setattr(control.asyncio, "to_thread", fake_to_thread)

    response = asyncio.run(control.api_update_check(None))

    assert response.status_code == 200
    assert calls == [(control._managed_update_payload, (), {"fetch": True, "include_tags": True})]


def test_update_apply_keeps_the_event_loop_responsive(monkeypatch):
    import supervisor.update_merge as update_merge

    started = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(update_merge, "active_update_tx", lambda: {})

    def blocked_apply(*_args, **_kwargs):
        started.set()
        assert release.wait(2)
        return JSONResponse({"status": "ok"})

    monkeypatch.setattr(control, "_apply_smart_update_fenced", blocked_apply)

    async def exercise():
        task = asyncio.create_task(control.api_update_apply(_Request({
            "strategy": "auto_merge",
            "expected_base_sha": BASE,
            "expected_target_sha": TARGET,
        })))
        while not started.is_set():
            await asyncio.sleep(0.001)
        await asyncio.sleep(0)
        release.set()
        return await task

    response = asyncio.run(exercise())
    assert _body(response) == {"status": "ok"}


def test_restore_keeps_the_event_loop_responsive(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def blocked_restore(*_args, **_kwargs):
        started.set()
        assert release.wait(2)
        return JSONResponse({"status": "ok"})

    monkeypatch.setattr(control, "_git_rollback_fenced", blocked_restore)

    async def exercise():
        task = asyncio.create_task(control.api_git_rollback(_Request({"target": "v1"})))
        while not started.is_set():
            await asyncio.sleep(0.001)
        await asyncio.sleep(0)
        release.set()
        return await task

    response = asyncio.run(exercise())
    assert _body(response) == {"status": "ok"}


def test_unknown_strategy_is_400_and_never_falls_back_to_replace():
    response = asyncio.run(control.api_update_apply(_Request({"strategy": "force"})))

    assert response.status_code == 400
    # ABI-3 ingress schema: the closed strategy vocabulary refuses at the
    # derived-schema gate before the bespoke check.
    assert "strategy must be one of" in _body(response)["error"]


def test_update_apply_rejects_non_object_json():
    response = asyncio.run(control.api_update_apply(_Request(["replace"])))

    assert response.status_code == 400
    assert "must be an object" in _body(response)["error"]


def test_replace_requires_explicit_recovery_confirmation(monkeypatch):
    import supervisor.update_merge as update_merge

    monkeypatch.setattr(update_merge, "active_update_tx", lambda: {})
    response = asyncio.run(control.api_update_apply(_Request({
        "strategy": "replace",
        "expected_base_sha": BASE,
        "expected_target_sha": TARGET,
    })))

    assert response.status_code == 400
    assert "confirm_recovery" in _body(response)["error"]


def test_apply_requires_exact_preflight_pins(monkeypatch):
    import supervisor.update_merge as update_merge

    monkeypatch.setattr(update_merge, "active_update_tx", lambda: {})
    response = asyncio.run(control.api_update_apply(_Request({"strategy": "auto_merge"})))

    assert response.status_code == 400
    assert "base and target SHA" in _body(response)["error"]


def _wire_smart(monkeypatch, plans):
    import supervisor.update_merge as update_merge

    iterator = iter(plans)
    monkeypatch.setattr(update_merge, "plan_managed_update_merge", lambda **_kwargs: next(iterator))
    monkeypatch.setattr(update_merge, "active_update_tx", lambda: {})
    monkeypatch.setattr(update_merge, "acquire_update_lock", lambda: object())
    monkeypatch.setattr(update_merge, "release_update_lock", lambda _lock: None)
    monkeypatch.setattr(control, "_quiesce_repo_writers", lambda _reason: [])
    # Stash-first prologue (Q9): stub the shared stash step and its unwind so
    # routing tests exercise ONLY the lane choice.
    monkeypatch.setattr(
        control,
        "_stash_local_work_fenced",
        lambda **_kwargs: ({"phase": "stashing_local_work", "stash_sha": "", "local_work_carrier": "none"}, None),
    )
    monkeypatch.setattr(control, "_unwind_stashed_update", lambda _tx, _context: "")


def test_smart_update_routes_clean_plan_to_supervisor_apply(monkeypatch):
    _wire_smart(monkeypatch, [_plan(merge_commit=""), _plan()])
    monkeypatch.setattr(
        control,
        "_apply_clean_merge_fenced",
        lambda _request, plan, _tx: JSONResponse({"route": "clean", "target": plan["target_sha"]}),
    )
    monkeypatch.setattr(
        control,
        "_start_assisted_merge_fenced",
        lambda *_args: (_ for _ in ()).throw(AssertionError("assisted should not run")),
    )

    response = asyncio.run(control._apply_smart_update(
        _Request(), expected_base_sha=BASE, expected_target_sha=TARGET
    ))

    assert _body(response) == {"route": "clean", "target": TARGET}


def test_smart_update_routes_clean_divergence_to_deterministic_git_path(monkeypatch):
    diverged = _plan(merge_commit="c" * 40)
    _wire_smart(monkeypatch, [diverged, diverged])
    monkeypatch.setattr(
        control,
        "_apply_clean_merge_fenced",
        lambda _request, plan, _tx: JSONResponse({"route": "clean", "merge": plan["merge_commit"]}),
    )
    monkeypatch.setattr(
        control,
        "_start_assisted_merge_fenced",
        lambda *_args: (_ for _ in ()).throw(AssertionError("LLM must not run without conflicts")),
    )

    response = asyncio.run(control._apply_smart_update(
        _Request(), expected_base_sha=BASE, expected_target_sha=TARGET
    ))

    assert _body(response) == {"route": "clean", "merge": "c" * 40}


def test_smart_update_routes_conflicting_plan_to_assisted(monkeypatch):
    """Stash-first supersession: the OLD contract routed a DIRTY plan to
    assisted; since Q9 the apply-path replan runs post-stash on a clean tree,
    so the assisted route is chosen by CONFLICTS alone (a dirty plan2 is the
    late-local-changes abort, pinned by its own test below)."""
    conflicting = _plan(
        kind="conflicting", merge_commit="", code_conflict_paths=["ouroboros/loop.py"]
    )
    _wire_smart(monkeypatch, [conflicting, conflicting])
    monkeypatch.setattr(
        control,
        "_apply_clean_merge_fenced",
        lambda *_args: (_ for _ in ()).throw(AssertionError("clean apply should not run")),
    )
    monkeypatch.setattr(
        control,
        "_start_assisted_merge_fenced",
        lambda plan, _tx: JSONResponse({"route": "assisted", "conflicts": plan["code_conflict_paths"]}),
    )

    response = asyncio.run(control._apply_smart_update(
        _Request(), expected_base_sha=BASE, expected_target_sha=TARGET
    ))

    assert _body(response) == {"route": "assisted", "conflicts": ["ouroboros/loop.py"]}


def test_assisted_update_refuses_before_mutation_when_budget_is_exhausted(monkeypatch):
    import supervisor.git_ops as git_ops
    import supervisor.state as state

    monkeypatch.setattr(state, "load_state", lambda: {})
    monkeypatch.setattr(state, "budget_remaining", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(control, "_respawn_workers_after_failed_update", lambda: None)
    monkeypatch.setattr(
        git_ops,
        "_create_rescue_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("budget refusal must happen before rescue/materialization")
        ),
    )

    response = control._start_assisted_merge_fenced(_plan(
        kind="conflicting",
        local_dirty_count=1,
        local_snapshot=BASE,
        code_conflict_paths=["local.py"],
    ), {"phase": "stashing_local_work", "stash_sha": "", "local_work_carrier": "none"})

    assert response.status_code == 409
    assert "needs model budget" in _body(response)["error"]


def test_assisted_update_refuses_when_one_review_wave_is_unaffordable(monkeypatch):
    """Affordability floor: remaining budget above zero but below one estimated
    triad+scope wave refuses BEFORE any repo mutation (rescue included)."""
    import ouroboros.reviewer_slot_config as reviewer_slot_config
    import ouroboros.usage_accounting as usage_accounting
    import supervisor.git_ops as git_ops
    import supervisor.state as state

    monkeypatch.setattr(state, "load_state", lambda: {})
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 0.4)
    monkeypatch.setattr(control, "_respawn_workers_after_failed_update", lambda: None)
    api_row = SimpleNamespace(target_id="openai/gpt-test", is_session=False)
    monkeypatch.setattr(reviewer_slot_config, "commit_triad_rows", lambda: [api_row])
    monkeypatch.setattr(reviewer_slot_config, "commit_scope_rows", lambda: [api_row])
    monkeypatch.setattr(
        usage_accounting,
        "review_wave_admission",
        lambda **kwargs: {
            "fits": False,
            "estimated_wave_usd": 3.21,
            "remaining_usd": kwargs.get("remaining_usd_override"),
        },
    )
    monkeypatch.setattr(
        git_ops,
        "_create_rescue_snapshot",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("floor refusal must happen before rescue/materialization")
        ),
    )

    response = control._start_assisted_merge_fenced(_plan(
        kind="conflicting",
        local_dirty_count=0,
        local_snapshot=BASE,
        code_conflict_paths=["local.py"],
    ), {"phase": "stashing_local_work", "stash_sha": "", "local_work_carrier": "none"})

    assert response.status_code == 409
    body = _body(response)
    assert "review wave" in body["error"]
    assert body["estimated_wave_usd"] == 6.42  # triad + scope surfaces summed


@pytest.mark.parametrize(("ready", "expected_status"), [(True, 200), (False, 409)])
def test_assisted_resolver_boots_before_conflicts_reach_live_tree(
    monkeypatch, ready, expected_status
):
    import ouroboros.reviewer_slot_config as reviewer_slot_config
    import supervisor.git_ops as git_ops
    import supervisor.state as state
    import supervisor.update_merge as update_merge
    import supervisor.workers as workers

    # No API reviewer rows -> the wave-floor estimator is skipped entirely
    # (agent-session rows ride subscriptions, not USD budget).
    monkeypatch.setattr(reviewer_slot_config, "commit_triad_rows", lambda: [])
    monkeypatch.setattr(reviewer_slot_config, "commit_scope_rows", lambda: [])
    calls = []
    monkeypatch.setattr(git_ops, "BRANCH_DEV", "ouroboros")
    monkeypatch.setattr(git_ops, "_create_rescue_snapshot", lambda *_a, **_k: None)
    monkeypatch.setattr(
        git_ops, "_collect_repo_sync_state",
        lambda: {
            "current_branch": "ouroboros", "dirty_lines": [],
            "unpushed_lines": [], "warnings": [],
        },
    )
    def _capture(cmd):
        if "--abbrev-ref" in cmd:
            return 0, "ouroboros", ""
        if "MERGE_HEAD" in cmd:
            return 1, "", ""
        if cmd[:3] == ["git", "rev-parse", "--verify"]:
            return 0, BASE, ""
        return 0, "", ""

    monkeypatch.setattr(git_ops, "git_capture", _capture)
    monkeypatch.setattr(state, "load_state", lambda: {"owner_chat_id": 1})
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 10.0)
    monkeypatch.setattr(update_merge, "create_rescue_local_ref", lambda _sha: True)
    monkeypatch.setattr(
        update_merge,
        "ensure_assisted_resolver_ready",
        lambda expected_sha: calls.append(f"resolver_ready:{expected_sha}") or ready,
    )
    monkeypatch.setattr(
        update_merge,
        "write_update_tx",
        lambda tx: calls.append(f"tx:{tx['phase']}"),
    )
    monkeypatch.setattr(
        update_merge,
        "materialize_assisted_merge_live",
        lambda *_a: (calls.append("materialize") or True, "ok", "m0tree"),
    )
    monkeypatch.setattr(
        update_merge,
        "enqueue_assisted_resolution_task",
        lambda _tx: calls.append("enqueue") or "resolver-task",
    )
    monkeypatch.setattr(
        workers,
        "close_repo_writer_admission",
        lambda _reason: calls.append("close_gate"),
    )
    monkeypatch.setattr(
        workers,
        "kill_workers_for_update",
        lambda **_kwargs: calls.append("kill_rejected_worker") or [],
    )
    monkeypatch.setattr(
        control,
        "_respawn_workers_after_failed_update",
        lambda: calls.append("respawn_clean_pool"),
    )

    response = control._start_assisted_merge_fenced(_plan(
        kind="conflicting", local_dirty_count=1,
        local_snapshot=BASE, code_conflict_paths=["ouroboros/config.py"],
    ), {"phase": "stashing_local_work", "stash_sha": "", "local_work_carrier": "none"})

    assert response.status_code == expected_status
    if ready:
        assert calls == [
            "close_gate", f"resolver_ready:{BASE}", "tx:materializing_assisted",
            "materialize", "tx:assisted_resolution", "enqueue",
        ]
    else:
        assert calls == [
            "close_gate", f"resolver_ready:{BASE}",
            "kill_rejected_worker", "respawn_clean_pool",
        ]


def test_stash_lookup_unknown_keeps_the_durable_tx_for_boot(monkeypatch):
    """A push whose entry exists but cannot be LISTED must keep the
    stashing_local_work tx on disk — boot retries the lookup and restores;
    clearing it would orphan the owner's work behind an HTTP error."""
    import supervisor.git_ops as git_ops
    import supervisor.update_merge as update_merge

    monkeypatch.setattr(git_ops, "git_capture", lambda cmd: (0, " M dirty.txt", ""))
    writes, cleared = [], []
    monkeypatch.setattr(update_merge, "write_update_tx", lambda tx: writes.append(dict(tx)))
    monkeypatch.setattr(
        update_merge, "stash_local_changes_for_update",
        lambda _attempt: ("lookup_unknown", "", "storage down"),
    )
    monkeypatch.setattr(
        update_merge, "clear_update_tx",
        lambda: cleared.append(True) or True,
    )
    monkeypatch.setattr(control, "_respawn_workers_after_failed_update", lambda: None)

    tx, failure = control._stash_local_work_fenced(
        branch="ouroboros", base_sha=BASE, target_sha=TARGET, plan=_plan(),
    )

    assert tx is None and failure is not None
    assert failure.status_code == 409
    assert _body(failure)["reason"] == "stash_lookup_unknown"
    assert writes, "the durable stashing_local_work tx was never written"
    assert cleared == [], "the tx was cleared — boot can no longer recover the stash"


def test_resolver_fence_blockers_still_unwind_the_stash(monkeypatch):
    """Hung writers after a failed resolver boot must not strand the owner's
    stashed work behind a live tx: the tree was never touched, so the prologue
    is unwound and the fence response discloses the restore."""
    import ouroboros.reviewer_slot_config as reviewer_slot_config
    import supervisor.git_ops as git_ops
    import supervisor.state as state
    import supervisor.update_merge as update_merge
    import supervisor.workers as workers

    monkeypatch.setattr(reviewer_slot_config, "commit_triad_rows", lambda: [])
    monkeypatch.setattr(reviewer_slot_config, "commit_scope_rows", lambda: [])
    monkeypatch.setattr(git_ops, "BRANCH_DEV", "ouroboros")
    monkeypatch.setattr(git_ops, "_create_rescue_snapshot", lambda *_a, **_k: None)
    monkeypatch.setattr(
        git_ops, "_collect_repo_sync_state",
        lambda: {"current_branch": "ouroboros", "dirty_lines": [], "unpushed_lines": [], "warnings": []},
    )
    monkeypatch.setattr(state, "load_state", lambda: {"owner_chat_id": 1})
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 10.0)
    monkeypatch.setattr(update_merge, "create_rescue_local_ref", lambda _sha: "rescue-local-x")
    monkeypatch.setattr(update_merge, "ensure_assisted_resolver_ready", lambda _sha: False)
    monkeypatch.setattr(workers, "close_repo_writer_admission", lambda _reason: None)
    monkeypatch.setattr(workers, "kill_workers_for_update", lambda **_k: ["stuck-writer"])
    unwound = []
    monkeypatch.setattr(
        control, "_unwind_stashed_update",
        lambda _tx, context: unwound.append(context) or "local changes restored",
    )

    response = control._start_assisted_merge_fenced(_plan(
        kind="conflicting", local_dirty_count=0,
        local_snapshot=BASE, code_conflict_paths=["x.py"],
    ), {"phase": "stashing_local_work", "stash_sha": "s" * 40, "local_work_carrier": "stash"})

    assert response.status_code == 409
    body = _body(response)
    assert body["reason"] == "update_writer_fence_blocked"
    assert body["stash_note"] == "local changes restored"
    assert unwound == ["assisted_resolver_fence_blocked"]


def test_unknown_plan_refuses_before_writer_fence(monkeypatch):
    import supervisor.update_merge as update_merge

    monkeypatch.setattr(
        update_merge,
        "plan_managed_update_merge",
        lambda **_kwargs: _plan(kind="unknown", error="status failed"),
    )
    monkeypatch.setattr(
        control,
        "_quiesce_repo_writers",
        lambda _reason: (_ for _ in ()).throw(AssertionError("must not stop workers")),
    )

    response = asyncio.run(control._apply_smart_update(
        _Request(), expected_base_sha=BASE, expected_target_sha=TARGET
    ))

    assert response.status_code == 409
    assert _body(response)["error"] == "status failed"


def test_post_fence_sha_drift_aborts_and_respawns(monkeypatch):
    _wire_smart(monkeypatch, [_plan(merge_commit=""), _plan(target_sha="c" * 40)])
    calls = []
    monkeypatch.setattr(control, "_respawn_workers_after_failed_update", lambda: calls.append("respawn"))

    response = asyncio.run(control._apply_smart_update(
        _Request(), expected_base_sha=BASE, expected_target_sha=TARGET
    ))

    assert response.status_code == 409
    assert _body(response)["reason"] == "release_moved"
    assert calls == ["respawn"]


def test_late_local_changes_after_stash_abort_the_update(monkeypatch):
    """The post-stash replan must describe a CLEAN tree: late dirt (a human
    editing between the stash and the replan) aborts fail-closed instead of
    riding the synthetic-snapshot path into committed history."""
    dirty_plan2 = _plan(local_dirty_count=2, merge_commit="")
    _wire_smart(monkeypatch, [_plan(merge_commit=""), dirty_plan2])
    calls = []
    monkeypatch.setattr(control, "_respawn_workers_after_failed_update", lambda: calls.append("respawn"))

    response = asyncio.run(control._apply_smart_update(
        _Request(), expected_base_sha=BASE, expected_target_sha=TARGET
    ))

    assert response.status_code == 409
    assert _body(response)["reason"] == "late_local_changes"
    assert calls == ["respawn"]


def test_landed_update_without_restart_callback_is_a_successful_manual_restart_response():
    request = _Request()
    request.app.state = SimpleNamespace()

    response = control._restart_response(request, strategy="auto_merge", plan=_plan())

    assert response.status_code == 200
    assert _body(response)["status"] == "restart_required"


def test_clean_apply_publishes_smoke_proof_only_after_pass(monkeypatch):
    import supervisor.git_ops as git_ops
    import supervisor.update_merge as update_merge

    writes = []
    monkeypatch.setattr(git_ops, "BRANCH_DEV", "ouroboros")
    monkeypatch.setattr(git_ops, "_create_rescue_snapshot", lambda *_a, **_k: None)
    monkeypatch.setattr(
        git_ops, "_collect_repo_sync_state",
        lambda: {
            "current_branch": "ouroboros", "dirty_lines": [],
            "unpushed_lines": [], "warnings": [],
        },
    )
    def _capture(cmd):
        if "--abbrev-ref" in cmd:
            return 0, "ouroboros", ""
        if "MERGE_HEAD" in cmd:
            return 1, "", ""
        if cmd[:3] == ["git", "rev-parse", "--verify"]:
            return 0, BASE, ""
        return 0, "", ""

    monkeypatch.setattr(git_ops, "git_capture", _capture)
    monkeypatch.setattr(update_merge, "write_update_tx", lambda tx: writes.append(dict(tx)))
    monkeypatch.setattr(update_merge, "apply_managed_merge_update", lambda *_a: (True, "ok"))
    monkeypatch.setattr(update_merge, "update_restart_smoke", lambda: {"ok": True})

    response = control._apply_clean_merge_fenced(
        _Request(), _plan(),
        {"phase": "stashing_local_work", "stash_sha": "", "local_work_carrier": "none",
         "pre_update_sha": BASE, "pre_update_branch": "ouroboros", "attempt_id": "attempt"},
    )

    assert response.status_code == 200
    assert [tx["pre_restart_smoke"] for tx in writes] == ["pending", "passed"]


def test_replace_apply_publishes_smoke_proof_only_after_pass(monkeypatch):
    import supervisor.git_ops as git_ops
    import supervisor.update_merge as update_merge

    writes = []
    plans = iter([_plan(), _plan()])
    monkeypatch.setattr(git_ops, "BRANCH_DEV", "ouroboros")
    monkeypatch.setattr(git_ops, "_write_update_intent", lambda _intent: None)
    monkeypatch.setattr(git_ops, "checkout_and_reset", lambda *_a, **_k: (True, "ok"))
    monkeypatch.setattr(
        git_ops,
        "prepare_managed_update",
        lambda *_a, **_k: (True, {"update_intent": {"target_sha": TARGET}}),
    )
    monkeypatch.setattr(update_merge, "plan_managed_update_merge", lambda **_k: next(plans))
    monkeypatch.setattr(update_merge, "acquire_update_lock", lambda: object())
    monkeypatch.setattr(update_merge, "release_update_lock", lambda _lock: None)
    monkeypatch.setattr(update_merge, "active_update_tx", lambda: {})
    monkeypatch.setattr(update_merge, "write_update_tx", lambda tx: writes.append(dict(tx)))
    monkeypatch.setattr(update_merge, "update_restart_smoke", lambda: {"ok": True})
    monkeypatch.setattr(control, "_quiesce_repo_writers", lambda _reason: [])

    response = control._apply_replace_recovery_fenced(
        _Request(),
        expected_base_sha=BASE,
        expected_target_sha=TARGET,
    )

    assert response.status_code == 200
    assert [tx["pre_restart_smoke"] for tx in writes] == ["pending", "pending", "passed"]


def test_writer_fence_order(monkeypatch):
    import ouroboros.process_custody as process_custody
    import ouroboros.tools.services as services
    import supervisor.workers as workers

    calls = []
    monkeypatch.setattr(workers, "close_repo_writer_admission", lambda _reason: calls.append("close"))
    monkeypatch.setattr(workers, "drain_repo_writers", lambda: calls.append("drain") or [])
    monkeypatch.setattr(
        workers,
        "kill_workers_for_update",
        lambda **_kwargs: calls.append("kill_workers") or [],
    )
    monkeypatch.setattr(
        services,
        "kill_all_services",
        lambda *_args, **_kwargs: calls.append("kill_services") or [],
    )
    # The custody sweep is the fence's fifth step and reads
    # supervisor.git_ops.DRIVE_ROOT. The shared pytest bootstrap now rebinds it
    # to a disposable root; before that binding, this order-only test reached the
    # operator's live process ledger: it reported live entries as blockers
    # (any machine running Ouroboros failed this test) and, worse, would have
    # killed a ledgered task/session service that happened to be running.
    monkeypatch.setattr(
        process_custody,
        "quiesce_custodied_services",
        lambda *_args, **_kwargs: (calls.append("quiesce_custody") or (True, [])),
    )

    assert control._quiesce_repo_writers("test") == []
    assert calls == ["close", "drain", "kill_workers", "kill_services", "quiesce_custody"]


def test_failed_direct_turn_drain_does_not_kill_pool(monkeypatch):
    import supervisor.workers as workers

    calls = []
    monkeypatch.setattr(workers, "close_repo_writer_admission", lambda _reason: calls.append("close"))
    monkeypatch.setattr(workers, "drain_repo_writers", lambda: calls.append("drain") or ["direct_chat"])
    monkeypatch.setattr(workers, "open_repo_writer_admission", lambda: calls.append("open"))
    monkeypatch.setattr(
        workers,
        "kill_workers_for_update",
        lambda **_kwargs: calls.append("kill") or [],
    )

    blockers = control._quiesce_repo_writers("test")

    assert blockers == ["active:direct_chat"]
    assert calls == ["close", "drain", "open"]


def test_durable_assisted_tx_keeps_pending_tasks_behind_the_writer_gate(
    monkeypatch, tmp_path
):
    import supervisor.git_ops as git_ops
    import supervisor.update_merge as update_merge
    import supervisor.workers as workers

    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setattr(git_ops, "_git_dir", lambda: repo / ".git")
    workers.open_repo_writer_admission()  # prove the durable marker is sufficient
    tx = {
        "phase": "assisted_resolution",
        "task_id": "resolver-task",
    }
    metadata = {
        "managed_update": {
            "authority_fingerprint": update_merge.assisted_authority_fingerprint(tx),
        }
    }
    update_merge.write_update_tx(tx)

    assert workers.repo_writer_admission_closed().startswith("managed_update_tx:")
    assert workers.repo_writer_task_allowed({"id": "ordinary-pending"}) is False
    assert workers.repo_writer_task_allowed({
        "id": "resolver-task",
        "metadata": metadata,
    }) is True

    assert update_merge.clear_update_tx() is True
    assert workers.repo_writer_admission_closed() == ""
    assert workers.repo_writer_task_allowed({"id": "ordinary-pending"}) is True


def test_assisted_task_end_reopens_only_its_cross_process_writer_gate(
    monkeypatch, tmp_path
):
    import supervisor.git_ops as git_ops
    import supervisor.update_merge as update_merge
    import supervisor.workers as workers

    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path / "data")
    monkeypatch.setattr(git_ops, "_git_dir", lambda: repo / ".git")
    tx = {"phase": "assisted_resolution", "task_id": "resolver-task"}
    metadata = {
        "managed_update": {
            "authority_fingerprint": update_merge.assisted_authority_fingerprint(tx),
        }
    }
    reason = update_merge.assisted_writer_gate_reason(tx)
    workers.close_repo_writer_admission(reason)

    assert update_merge.release_assisted_writer_gate_after_task({}) is False
    assert workers.repo_writer_admission_closed() == reason
    update_merge.write_update_tx(tx)
    assert update_merge.release_assisted_writer_gate_after_task(metadata) is False
    assert workers.repo_writer_admission_closed() == reason
    assert update_merge.clear_update_tx() is True
    assert update_merge.release_assisted_writer_gate_after_task(metadata) is True
    assert workers.repo_writer_admission_closed() == ""
