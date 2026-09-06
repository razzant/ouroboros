"""Preflight recovery uses the existing canonical delegate request and one row."""

import json

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


@pytest.mark.parametrize("end", ["unresolved", "crash", "success"])
def test_native_preflight_uses_existing_physical_stamp(preflight, monkeypatch, end):
    from ouroboros.review_execution import ReviewAttemptResult
    from ouroboros.usage_accounting import PhysicalAttemptCapture

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

    class Executor:
        def __init__(self, assignment, **kwargs):
            self.assignment = assignment

        def execute(self):
            self.assignment.dispatch_stamp()
            pending = load_state(preflight.drive_root).advisory_runs[-1]
            assert pending.status == "pending"
            assert pending.execution["operation_id"] == self.assignment.call_id
            sent.append(self.assignment.call_id)
            if end == "crash":
                raise Crash()
            if end == "unresolved":
                error = RuntimeError("provider outcome unknown")
                error.physical_attempt_capture = PhysicalAttemptCapture(
                    attempt_id="native-attempt", model="openai/test", provider="openai", state="unresolved", candidate_measurement_kind="canonical_json_v1",
                )
                raise error
            return ReviewAttemptResult(message={"content": "[]", "native_transcript": "```json\n[]\n```"}, usage={"native_rounds": 1}, raw_text="[]")

        def failure_custody(self):
            return {"native_rounds": 1, "native_terminal_round": "retained reviewer note"}

    monkeypatch.setattr("ouroboros.review_native_episode.NativeToolRoundReviewExecutor", Executor)
    if end == "crash":
        with pytest.raises(Crash):
            _run(preflight)
    else:
        assert _run(preflight)["status"] == ("fresh" if end == "success" else "error")
    rows = load_state(preflight.drive_root).advisory_runs
    assert len(rows) == 1
    if end == "success":
        assert rows[0].execution["source_text"] == "```json\n[]\n```"
    if end == "unresolved":
        assert rows[0].execution["usage"]["native_terminal_round"] == "retained reviewer note"
    if end != "success":
        assert _run(preflight)["status"] == "pending"
    assert len(sent) == 1


def test_preflight_preserves_source_before_verdict_canonicalization(preflight):
    source = '{"findings": []}'
    FakeGateway.detail["primaryOutput"]["text"] = source
    FakeGateway.detail["summary"]["outputConformance"] = "passed"
    assert _run(preflight)["status"] == "fresh"
    row = load_state(preflight.drive_root).advisory_runs[-1]
    assert row.raw_result == "[]"
    assert row.execution["source_text"] == source
