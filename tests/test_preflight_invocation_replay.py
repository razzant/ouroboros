"""Preflight recovery uses the existing canonical delegate request and one row."""

import json
import os

import pytest

from ouroboros import delegate_custody
from ouroboros.review_state import load_state
from ouroboros.tools import claude_advisory_review as advisory
from tests.test_advisory_inline_freshness import candidate  # noqa: F401
from tests._review_session_route_shared import (
    FakeGateway,
    _owned_gateway_uses_each_test_transport,  # noqa: F401
    fake_route,  # noqa: F401
)


@pytest.fixture
def preflight(candidate, fake_route, monkeypatch):  # noqa: F811 - imported pytest fixtures
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", json.dumps({
        "triad": [{"slot_id": "t", "route": {"kind": "api_chat", "target_id": "openai/test"}}],
        "scope": [{"slot_id": "s", "route": {"kind": "api_chat", "target_id": "openai/test"}}],
        "advisory": {"enabled": True, "route": {"kind": "agent_session", "target_id": "fake-review=fake-small"}},
    }))
    monkeypatch.setattr(advisory, "_run_advisory_tests", lambda ctx: None)
    monkeypatch.setattr(advisory, "_llm_extract_advisory_items", lambda *a: pytest.fail("clean fixture needs no paid extraction"))
    return candidate


def _run(ctx):
    return json.loads(advisory._handle_advisory_pre_review(ctx, "candidate", paths=["change.py"], review_rebuttal="evidence", prepared=True))


def _posts():
    return [(key, body) for gateway in FakeGateway.instances for key, body in zip(gateway.start_keys, gateway.start_requests)]


def test_unknown_start_replays_same_key_and_canonical_body(preflight):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    FakeGateway.start_error = ClaudexorUnavailable("daemon_unreachable", "accepted response lost", status_code=0)
    first = _run(preflight)
    assert first["status"] == "error"
    before = load_state(preflight.drive_root).advisory_runs[-1]
    token = before.execution["pending_invocation_id"]
    assert token and len(_posts()) == 1
    stored = delegate_custody.invocation_record(preflight.drive_root, token)
    assert stored["request"] == _posts()[0][1]
    second = _run(preflight)
    assert second["status"] == "fresh"
    assert len(_posts()) == 2 and _posts()[0] == _posts()[1]
    rows = load_state(preflight.drive_root).advisory_runs
    assert len(rows) == 1 and rows[0].attempt == before.attempt
    assert rows[0].execution["invocation_id"] == token
    assert rows[0].execution["pending_invocation_id"] == ""
    assert rows[0].review_rebuttal == "evidence"


def test_crash_after_terminal_before_logical_save_joins_run_without_post(preflight, monkeypatch):
    real_update = advisory.update_state

    def crash_on_terminal(drive, mutate):
        def crash(state):
            value = mutate(state)
            if state.advisory_runs and state.advisory_runs[-1].status == "fresh":
                raise RuntimeError("crash before terminal logical record")
            return value
        return real_update(drive, crash)

    monkeypatch.setattr(advisory, "update_state", crash_on_terminal)
    with pytest.raises(RuntimeError, match="crash before terminal"):
        _run(preflight)
    assert len(_posts()) == 1
    pending = load_state(preflight.drive_root).advisory_runs[-1]
    assert pending.execution["pending_invocation_id"]
    monkeypatch.setattr(advisory, "update_state", real_update)
    monkeypatch.setattr(advisory, "_run_advisory_tests", lambda ctx: pytest.fail("exact pending recovery does not rerun tests"))
    assert _run(preflight)["status"] == "fresh"
    assert len(_posts()) == 1
    assert len(load_state(preflight.drive_root).advisory_runs) == 1


@pytest.mark.parametrize("change", ["worktree", "index", "owner", "intent"])
def test_pending_mismatch_never_prepares_or_posts(preflight, monkeypatch, change):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from tests.test_review_prepared_candidate import _git

    FakeGateway.start_error = ClaudexorUnavailable("daemon_unreachable", "unknown", status_code=0)
    assert _run(preflight)["status"] == "error"
    if change == "worktree":
        (preflight.repo_dir / "change.py").write_text("value = 3\n")
    elif change == "index":
        _git(preflight, "reset", "HEAD")
    elif change == "owner":
        preflight.task_id = "foreign-task"
    before = _git(preflight, "write-tree")
    monkeypatch.setattr(advisory, "_auto_sync_release_metadata_if_needed", lambda *a: pytest.fail("must not prepare"))
    if change == "intent":
        result = json.loads(advisory._handle_advisory_pre_review(preflight, "different intent", paths=["change.py"]))
    else:
        result = _run(preflight)
    assert result["status"] == "pending"
    assert {"worktree": "snapshot_hash", "index": "staged_fingerprint", "owner": "task_id",
            "intent": "intent.commit_message"}[change] in result["error"]
    assert len(_posts()) == 1
    assert _git(preflight, "write-tree") == before


def test_failed_logical_checkpoint_prevents_provider_post(preflight, monkeypatch):
    persist = advisory._persist_preflight_record

    def fail_pending(ctx, snapshot_hash, commit_message, record):
        if record.get("status") == "pending":
            raise OSError("logical checkpoint disk failure")
        return persist(ctx, snapshot_hash, commit_message, record)

    monkeypatch.setattr(advisory, "_persist_preflight_record", fail_pending)
    assert _run(preflight)["status"] == "error"
    assert _posts() == []


@pytest.mark.parametrize("new_request", [False, True])
def test_audited_skip_releases_admission_but_preserves_unknown_run_and_late_source(preflight, new_request):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.tools.git_review_cycle import _reconcile_advisory_before_preparation

    FakeGateway.start_error = ClaudexorUnavailable("daemon_unreachable", "accepted response lost", status_code=0)
    assert _run(preflight)["status"] == "error"
    original = load_state(preflight.drive_root).advisory_runs[0]
    token = original.execution["pending_invocation_id"]
    canonical = delegate_custody.invocation_record(preflight.drive_root, token)
    skipped = json.loads(advisory._handle_advisory_pre_review(
        preflight, "new intent", paths=["change.py"], prepared=True, skip_advisory_review=True))
    assert skipped["status"] == "bypassed" and len(_posts()) == 1
    rows = load_state(preflight.drive_root).advisory_runs
    assert len(rows) == 2 and rows[0].execution == original.execution
    assert rows[0].raw_result == original.raw_result and rows[0].execution_pending
    assert rows[0].bypass_reason and not rows[0].blocks_preflight
    assert delegate_custody.invocation_record(preflight.drive_root, token) == canonical
    assert _reconcile_advisory_before_preparation(
        preflight, "new intent", goal="", scope="", paths=["change.py"], review_rebuttal="") == ""
    if new_request:
        covered = json.loads(advisory._handle_advisory_pre_review(
            preflight, "new intent", paths=["change.py"], prepared=False))
        assert covered["status"] == "already_fresh" and len(_posts()) == 1
        assert "bypassed" in covered["message"]
        fresh = json.loads(advisory._handle_advisory_pre_review(
            preflight, "new intent", paths=["change.py"], review_rebuttal="new evidence", prepared=False))
        assert fresh["status"] == "fresh" and len(_posts()) == 2
        assert _posts()[0][0] != _posts()[1][0]
        assert delegate_custody.invocation_record(preflight.drive_root, token) == canonical
        assert load_state(preflight.drive_root).advisory_runs[0].execution_pending
    # Exact old rejoin still works, with its original token, after logical skip.
    assert _run(preflight)["status"] == "fresh"
    assert len(_posts()) == 2 + int(new_request) and _posts()[0] == _posts()[-1]
    after = load_state(preflight.drive_root).advisory_runs
    assert len(after) == 2 + int(new_request)
    assert after[-1].status == ("fresh" if new_request else "bypassed")
    assert after[-1].commit_message == "new intent"
    assert after[0].execution["invocation_id"] == token and after[0].bypass_reason
    assert not after[0].execution_pending


def test_non_committing_cycle_forwards_skip_without_reposting_unknown_work(preflight, monkeypatch):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.tools import git

    FakeGateway.start_error = ClaudexorUnavailable("daemon_unreachable", "accepted response lost", status_code=0)
    assert _run(preflight)["status"] == "error"
    reached = []
    def stage(*args, **kwargs):
        reached.append(kwargs["skip_advisory_pre_review"])
        return {"status": "blocked", "block_reason": "controlled_stage"}
    monkeypatch.setattr(git, "_run_reviewed_stage_cycle", stage)
    result = git._run_non_committing_review_cycle(
        preflight, "new intent", paths=["change.py"], skip_advisory_review=True)
    assert reached == [True] and result["block_reason"] == "controlled_stage"
    assert len(_posts()) == 1 and load_state(preflight.drive_root).advisory_runs[0].execution_pending


def test_definite_start_failure_discharges_stranded_logical_checkpoint(preflight, monkeypatch):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    original_persist = advisory._persist_preflight_record

    def crash_before_error_save(ctx, snapshot_hash, commit_message, record):
        if record.get("status") != "pending":
            raise RuntimeError("lost logical terminal save")
        return original_persist(ctx, snapshot_hash, commit_message, record)

    FakeGateway.start_error = ClaudexorUnavailable("denied", "definite refusal", status_code=403)
    with monkeypatch.context() as fault:
        fault.setattr(advisory, "_persist_preflight_record", crash_before_error_save)
        with pytest.raises(RuntimeError, match="lost logical terminal save"):
            _run(preflight)
    old = load_state(preflight.drive_root).advisory_runs[0]
    token = old.execution["pending_invocation_id"]
    assert delegate_custody.invocation_record(preflight.drive_root, token)["state"] == "failed_definite"
    assert len(_posts()) == 1
    result = json.loads(advisory._handle_advisory_pre_review(
        preflight, "different intent", paths=["change.py"], prepared=True))
    assert result["status"] == "fresh"
    assert len(_posts()) == 2 and _posts()[0][0] != _posts()[1][0]
    rows = load_state(preflight.drive_root).advisory_runs
    assert rows[0].execution["invocation_id"] == token and not rows[0].execution_pending


@pytest.mark.parametrize("shape", ["execution", "container"])
def test_unreadable_pending_authority_never_dispatches(preflight, shape):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    FakeGateway.start_error = ClaudexorUnavailable("daemon_unreachable", "unknown", status_code=0)
    assert _run(preflight)["status"] == "error"
    path = preflight.drive_root / "state" / "advisory_review.json"
    state = json.loads(path.read_text())
    if shape == "execution":
        state["advisory_runs"][0]["execution"]["pending_invocation_id"] = []
    else:
        state["advisory_runs"] = {"hidden": state["advisory_runs"][0]}
    path.write_text(json.dumps(state))
    before = path.read_bytes()
    assert _run(preflight)["status"] == "pending"
    assert len(_posts()) == 1 and path.read_bytes() == before


def test_pending_invocation_survives_terminal_history_trim(preflight):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.review_state import AdvisoryRunRecord, update_state

    FakeGateway.start_error = ClaudexorUnavailable("daemon_unreachable", "unknown", status_code=0)
    assert _run(preflight)["status"] == "error"
    token = load_state(preflight.drive_root).advisory_runs[0].execution["pending_invocation_id"]
    for index in range(12):
        update_state(preflight.drive_root, lambda state: state.add_run(AdvisoryRunRecord(
            snapshot_hash=str(index), commit_message="other repo", status="fresh", ts=str(index), repo_key=f"/fixture-repo-{index}",
        )))
    retained = [row for row in load_state(preflight.drive_root).advisory_runs if row.execution.get("pending_invocation_id")]
    assert len(retained) == 1 and retained[0].execution["pending_invocation_id"] == token
    assert _run(preflight)["status"] == "fresh"
    assert _posts()[0] == _posts()[1]


def test_commit_entry_rejoins_pending_preflight_before_checkout(preflight, monkeypatch):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.tools import git
    from tests.test_review_prepared_candidate import _git

    FakeGateway.start_error = ClaudexorUnavailable("daemon_unreachable", "unknown", status_code=0)
    assert _run(preflight)["status"] == "error"
    before = _git(preflight, "write-tree")
    FakeGateway.start_error = ClaudexorUnavailable("daemon_unreachable", "still unknown", status_code=0)
    monkeypatch.setattr(git, "_prepare_review_commit_worktree", lambda *a, **kw: pytest.fail("must not checkout a pending candidate"))
    result = git._repo_commit_push(preflight, "candidate", paths=["change.py"], review_rebuttal="evidence")
    assert "REVIEW_PENDING" in result
    assert _git(preflight, "write-tree") == before
    assert len(_posts()) == 2 and _posts()[0] == _posts()[1]


def test_commit_explicit_skip_reaches_preparation_without_reposting_unknown_preflight(preflight, monkeypatch):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.tools import git
    from tests.test_review_prepared_candidate import _git

    FakeGateway.start_error = ClaudexorUnavailable("daemon_unreachable", "unknown", status_code=0)
    assert _run(preflight)["status"] == "error"
    before = _git(preflight, "write-tree")
    prepared = []
    monkeypatch.setattr(git, "_prepare_review_commit_worktree",
                        lambda *a, **kw: (prepared.append(True) or False, "controlled preparation boundary"))
    result = git._repo_commit_push(preflight, "changed intent", paths=["change.py"], skip_advisory_review=True)
    assert result == "controlled preparation boundary" and prepared == [True]
    assert len(_posts()) == 1 and _git(preflight, "write-tree") == before


@pytest.mark.parametrize("end", ["unresolved", "crash", "success"])
def test_native_preflight_uses_existing_episode_and_monetary_custody(preflight, monkeypatch, end):
    from ouroboros import usage_accounting as usage
    from ouroboros.reviewer_window import ReviewerWindow

    class Crash(BaseException):
        pass

    sent = []
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", json.dumps({
        "triad": [{"slot_id": "t", "route": {"kind": "api_chat", "target_id": "openai/test"}}],
        "scope": [{"slot_id": "s", "route": {"kind": "api_chat", "target_id": "openai/test"}}],
        "advisory": {"enabled": True, "route": {"kind": "api_chat", "target_id": "openai/test"}},
    }))
    monkeypatch.setattr(advisory, "advisory_review_route", lambda: "api_chat")
    monkeypatch.setattr("ouroboros.provider_models.model_has_credentials", lambda model: True)
    monkeypatch.setattr("ouroboros.reviewer_window.resolve_reviewer_window",
                        lambda model, **kw: ReviewerWindow(window_tokens=1_000_000, status="confirmed", model=model))

    class ControlledChat:
        def chat(self, **kwargs):
            # Native dispatch must not create a second advisory lifetime owner.
            assert not load_state(preflight.drive_root).advisory_runs
            sent.append(kwargs)
            if end != "success":
                reservation = usage.reserve_attempt(usage.AttemptRequest(
                    model="openai/test", provider="openai", reservation_usd=1.0,
                    drive_root=preflight.drive_root, task_id=preflight.task_id,
                    root_task_id=preflight.task_id, global_limit_usd=10.0,
                ))
                usage.mark_dispatched(reservation)
                usage.mark_unresolved(reservation, "controlled native outcome unknown")
                error = Crash() if end == "crash" else RuntimeError("provider outcome unknown")
                error.physical_attempt_capture = usage.PhysicalAttemptCapture(
                    attempt_id=reservation.attempt_id, model="openai/test", provider="openai",
                    state="unresolved", candidate_measurement_kind="canonical_json_v1")
                raise error
            return {"content": "[]"}, {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.0}

    monkeypatch.setattr("ouroboros.llm.LLMClient", ControlledChat)
    if end == "crash":
        with pytest.raises(Crash):
            _run(preflight)
    else:
        assert _run(preflight)["status"] == ("fresh" if end == "success" else "error")
    rows = load_state(preflight.drive_root).advisory_runs
    events = [json.loads(line) for line in delegate_custody.event_log_path(preflight.drive_root).read_text().splitlines()]
    episodes = [row for row in events if row.get("type") == "review_native_episode"]
    assert len(episodes) == 1 and episodes[0]["native_rounds"] == 1
    assert episodes[0]["task_id"] == preflight.task_id
    assert len(rows) == (0 if end == "crash" else 1)
    if end == "success":
        assert rows[0].execution["usage"]["native_custody_row"] == "written"
    if end == "unresolved":
        assert rows[0].execution["operation_state"] == "custody_lost"
        assert rows[0].execution["failure_code"] == "provider_outcome_unknown"
        assert _run(preflight)["status"] == "pending"
        skipped = json.loads(advisory._handle_advisory_pre_review(
            preflight, "different intent", paths=["change.py"], prepared=True, skip_advisory_review=True))
        assert skipped["status"] == "bypassed"
    if end != "success":
        projection = usage.usage_projection(preflight.drive_root, global_limit_usd=10.0)
        assert projection["accounted_usd"] == 1.0 and projection["remaining_known_usd"] == 9.0
        if rows:
            retained = load_state(preflight.drive_root).advisory_runs[0]
            assert retained.execution["operation_state"] == "custody_lost"
            assert retained.execution["usage"]["physical_attempt_state"] == "unresolved"
    assert len(sent) == 1


def test_preflight_preserves_source_before_verdict_canonicalization(preflight):
    source = '{"findings": []}'
    FakeGateway.detail["primaryOutput"]["text"] = source
    FakeGateway.detail["summary"]["outputConformance"] = "passed"
    assert _run(preflight)["status"] == "fresh"
    row = load_state(preflight.drive_root).advisory_runs[-1]
    assert row.raw_result == "[]"
    assert row.execution["source_text"] == source


@pytest.mark.parametrize("interrupted", [False, True])
def test_review_only_preserves_first_pending_index_before_local_meta(preflight, interrupted):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.tools import git
    from tests.test_review_prepared_candidate import _git

    class Crash(BaseException):
        pass

    before = _git(preflight, "write-tree")
    FakeGateway.start_error = Crash("interrupted after checkpoint") if interrupted else ClaudexorUnavailable(
        "daemon_unreachable", "accepted response lost", status_code=0)
    if interrupted:
        with pytest.raises(Crash):
            git._run_non_committing_review_cycle(preflight, "candidate", paths=["change.py"], review_rebuttal="evidence")
        assert preflight._last_claude_advisory_meta == {}
    else:
        result = git._run_non_committing_review_cycle(preflight, "candidate", paths=["change.py"], review_rebuttal="evidence")
        assert result["status"] == "blocked"
    row = load_state(preflight.drive_root).advisory_runs[-1]
    assert row.execution_pending and row.execution["pending_invocation_id"]
    assert _git(preflight, "write-tree") == before
    assert _git(preflight, "diff", "--cached", "--name-only") == "change.py"
    assert len(_posts()) == 1
    # A real rejoin must still be possible from the preserved candidate.
    assert _run(preflight)["status"] == "fresh"
    assert len(_posts()) == 2 and _posts()[0] == _posts()[1]


def test_completed_review_only_still_unstages(preflight, monkeypatch):
    from ouroboros.tools import git
    from tests.test_review_prepared_candidate import _git

    monkeypatch.setattr(git, "_run_parallel_review", lambda *a, **kw: (None, None, "", []))
    result = git._run_non_committing_review_cycle(preflight, "candidate", paths=["change.py"], review_rebuttal="evidence")
    assert result["status"] == "passed"
    assert _git(preflight, "diff", "--cached", "--name-only") == ""
    assert (preflight.repo_dir / "change.py").read_text() == "value = 2\n"
    assert not load_state(preflight.drive_root).advisory_runs[-1].execution_pending


@pytest.mark.parametrize("change", ["unchanged", "model", "profile", "kind"])
def test_pending_route_rejoins_canonical_session_or_refuses_changed_kind(preflight, monkeypatch, change):
    from dataclasses import asdict
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    FakeGateway.start_error = ClaudexorUnavailable("daemon_unreachable", "accepted response lost", status_code=0)
    assert _run(preflight)["status"] == "error"
    row = load_state(preflight.drive_root).advisory_runs[-1]
    token = row.execution["pending_invocation_id"]
    canonical = delegate_custody.invocation_record(preflight.drive_root, token)
    slots = json.loads(os.environ["OUROBOROS_REVIEWER_SLOTS"])
    route = slots["advisory"]["route"]
    if change == "model":
        route["target_id"] = "fake-review=different-model"
    elif change == "profile":
        route["profile_id"] = "different-profile"
    elif change == "kind":
        slots["advisory"]["route"] = {"kind": "api_chat", "target_id": "openai/test"}
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", json.dumps(slots))
    monkeypatch.setattr("ouroboros.provider_models.model_has_credentials", lambda _model: True)
    monkeypatch.setattr("ouroboros.review_native_episode.NativeToolRoundReviewExecutor",
                        lambda *a, **kw: pytest.fail("pending session must not dispatch native work"))
    result = _run(preflight)
    if change == "kind":
        assert result["status"] == "pending"
        assert result["failure_code"] == "advisory_pending_route_mismatch"
        assert len(_posts()) == 1
        assert delegate_custody.invocation_record(preflight.drive_root, token) == canonical
        assert asdict(load_state(preflight.drive_root).advisory_runs[-1]) == asdict(row)
    else:
        assert result["status"] == "fresh"
        assert len(_posts()) == 2 and _posts()[0] == _posts()[1]


def test_pending_refusal_names_the_changed_intent_and_exposes_existing_rejoin_inputs(preflight):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    FakeGateway.start_error = ClaudexorUnavailable("daemon_unreachable", "accepted response lost", status_code=0)
    assert _run(preflight)["status"] == "error"
    result = json.loads(advisory._handle_advisory_pre_review(
        preflight, "candidate", goal="different goal", paths=["change.py"], review_rebuttal="evidence"))
    assert result["status"] == "pending" and result["failure_code"] == "advisory_pending_mismatch"
    assert "intent.goal" in result["error"] and "different goal" not in result["error"]
    status = json.loads(advisory._handle_review_status(preflight))
    execution = status["advisory_runs"][0]["execution"]
    assert execution["intent"] == {"commit_message": "candidate", "goal": "", "scope": "", "review_rebuttal": "evidence"}
    assert execution["pending_invocation_id"] == load_state(preflight.drive_root).advisory_runs[-1].execution["pending_invocation_id"]
    assert len(_posts()) == 1
