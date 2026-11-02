"""Reviewer seats on the shared session transport: configured rights, private copies,
recorded-shape recovery and the raw-request replay a collection sends (no rebuilt prompt).

Synthetic transport (``FakeGateway``): no live engine, no provider, no money.
"""
from __future__ import annotations

import pathlib
import subprocess

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import review_checking
from ouroboros.review_execution import (
    ReviewRouteUnavailable,
    SessionInvocation,
    review_session_route,
    run_delegated_review_session,
)
from tests._review_session_route_shared import (  # noqa: F401 - imported pytest fixtures
    FakeGateway,
    _owned_gateway_uses_each_test_transport,
    fake_route,
)


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for args in (("init", "-q"), ("config", "user.email", "a@b"), ("config", "user.name", "a")):
        _git(root, *args)
    (root / "f.txt").write_text("v0\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")
    (root / "f.txt").write_text("A staged\n")
    _git(root, "add", "f.txt")
    (root / "f.txt").write_text("A staged\nB unstaged\n")
    return root


@pytest.fixture
def writable_engine(fake_route):  # noqa: F811
    FakeGateway.catalog_entry["accessProfilesSupported"] = ["readonly", "workspace_write", "full"]
    original = FakeGateway.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.engine_version = "3.22.0"
        self.trust = []
        self.ensure_full_access = lambda root: self.trust.append(root) or {"allowFullAccess": True}

    FakeGateway.__init__ = init
    yield
    FakeGateway.__init__ = original


def _invocation(**overrides):
    fields = dict(task_id="task-1", surface="multi_model_review", slot_id="t1", timeout_sec=60.0,
                  operation_id="op-1", session_route=review_session_route(), retry_state={})
    return SessionInvocation(**{**fields, **overrides})


def test_a_writable_seat_runs_in_its_own_copy_and_leaves_evidence(repo, tmp_path, writable_engine):
    drive = tmp_path / "drive"
    source = review_checking.freeze_checking_source(repo, data_root=drive, subject=review_checking.SUBJECT_INDEX)
    facts = run_delegated_review_session(prompt="review", root=str(repo), custody_drive=drive,
                                         invocation=_invocation(access="full", checking_source=source))
    body = FakeGateway.instances[-1].start_requests[0]
    area = review_checking.seat_dir(drive, "task-1", "op-1") / "area.noindex"
    assert body["mode"] == "agent" and body["access"] == "full" and body["scope"]["root"] == str(repo)
    assert body["execution"] == {"isolation": "live", "delegated": True, "workspaceRoot": str(area)}
    assert "CHECKING COPY" in body["instructions"] and source["tree"] in body["instructions"]
    assert FakeGateway.instances[-1].trust == [str(repo)]
    row = custody.invocation_record(drive, FakeGateway.instances[-1].start_keys[0])
    assert row["authority_source"] == "investigation" and row["execution_root"] == str(area)
    assert facts["requested_access"] == "full" and facts["checking_evidence"]["status"] == "retained"
    assert not area.exists()  # the copy goes once its receipt is written
    assert pathlib.Path(facts["checking_evidence"]["evidence_dir"], "summary.json").is_file()
    assert (repo / "f.txt").read_text() == "A staged\nB unstaged\n" and _git(repo, "write-tree") == source["tree"]


def test_a_read_only_seat_keeps_the_exact_ask_request(repo, tmp_path, fake_route):  # noqa: F811
    run_delegated_review_session(prompt="review", root=str(repo), custody_drive=tmp_path / "drive",
                                 invocation=_invocation(access="readonly"))
    body = FakeGateway.instances[-1].start_requests[0]
    assert (body["mode"], body["access"]) == ("ask", "readonly") and "execution" not in body
    assert "CHECKING COPY" not in body["instructions"]


@pytest.mark.parametrize("missing", ["source", "engine"])
def test_a_writable_seat_refuses_typed_before_any_send(repo, tmp_path, writable_engine, missing, monkeypatch):
    source = review_checking.freeze_checking_source(repo, data_root=tmp_path, subject=review_checking.SUBJECT_INDEX)
    stamps = []
    invocation = _invocation(access="workspace_write", checking_source=None if missing == "source" else source,
                             dispatch_stamp=lambda: stamps.append(1))
    if missing == "engine":
        original = FakeGateway.__init__

        def init(self, *a, **k):
            original(self, *a, **k)
            self.engine_version = "3.3.7"  # below the delegated workspaceRoot floor

        monkeypatch.setattr(FakeGateway, "__init__", init)
    with pytest.raises(ReviewRouteUnavailable) as caught:
        run_delegated_review_session(prompt="review", root=str(repo), custody_drive=tmp_path / "drive",
                                     invocation=invocation)
    assert caught.value.code == ("review_checking_source_missing" if missing == "source"
                                 else "review_checking_root_unsupported")
    assert not FakeGateway.instances[-1].start_requests and not stamps


def test_recovery_replays_the_recorded_rights_not_todays_configuration(repo, tmp_path, writable_engine):
    drive = tmp_path / "drive"
    source = review_checking.freeze_checking_source(repo, data_root=drive, subject=review_checking.SUBJECT_INDEX)
    FakeGateway.start_error = TimeoutError("accepted response lost")  # an unknown start outcome
    state = {}
    with pytest.raises(TimeoutError):
        run_delegated_review_session(prompt="review", root=str(repo), custody_drive=drive,
                                     invocation=_invocation(access="workspace_write", checking_source=source,
                                                            retry_state=state))
    token = state["pending_invocation_id"]
    first = FakeGateway.instances[-1].start_requests[0]
    # The row was lowered to read-only since: the pending invocation still replays as recorded.
    run_delegated_review_session(prompt="review", root=str(repo), custody_drive=drive,
                                 invocation=_invocation(access="readonly", retry_state=state))
    replay = FakeGateway.instances[-1]
    assert replay.start_keys == [token] and replay.start_requests[0] == first
    assert replay.start_requests[0]["access"] == "workspace_write"


def test_a_collection_sends_the_raw_recorded_request_never_a_rebuilt_prompt(tmp_path, fake_route):  # noqa: F811
    """A reconcile-only rejoin of a pending invocation replays its RAW START_REQUESTED body,
    even when the prompt the caller would rebuild differs and the original carried
    secret-shaped text the redacted prompt record could not reproduce."""
    from types import SimpleNamespace

    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_records import ReviewRequest, ReviewSlot
    from ouroboros.review_substrate import run_review_request

    drive = tmp_path / "drive"
    stored = {"prompt": "ORIGINAL review task password=hunter2 token=abc", "instructions": "i",
              "authPreference": "subscription", "mode": "ask", "access": "readonly",
              "scope": {"kind": "project", "root": str(tmp_path)}, "harnesses": ["fake-review"],
              "primaryHarness": "fake-review", "maxSeconds": 60}
    custody.record_start_requested(drive, run_id="", task_id="task-1", idempotency_key="k", invocation_id="inv-1",
                                   operation_id="op-1", max_seconds=60, request=stored, project_id="proj-1",
                                   project_owned=False, route="fake-review", surface="skill_review", slot_id="t1",
                                   source="review_substrate:skill_review")
    request = ReviewRequest(surface="skill_review", goal="g", task_id="task-1", retry_key="skill_review:abc",
                            session_task="A REBUILT prompt with a different advisory section",
                            session_root=str(tmp_path), reconcile_only=True)
    slot = ReviewSlot(slot_id="t1", model="fake-review=fake-small", route=ReviewRouteKind.AGENT_SESSION,
                      session_target="fake-review=fake-small")
    ctx = SimpleNamespace(drive_root=str(drive), task_id="task-1", _review_frozen_rows={"skill_review": {"t1": {
        "slot_id": "t1", "operation_id": "op-1", "pending_invocation_id": "inv-1",
        "operation_state": "in_flight", "late_result_pending": True}}})
    result = run_review_request(request, slots=[slot], drive_root=drive, usage_ctx=ctx)
    assert FakeGateway.instances, result.actors
    gateway = FakeGateway.instances[-1]
    assert gateway.start_keys == ["inv-1"] and gateway.start_requests[0]["prompt"] == stored["prompt"]
    assert result.actors[0]["status"] == "ok"
    with pytest.raises(FileNotFoundError):  # the rejoin wrote no replacement prompt record
        from ouroboros.observability import read_call_payload
        read_call_payload(drive, task_id="task-1", call_id="op-1_prompt")


def test_recorded_review_shapes(tmp_path):
    from ouroboros.review_session_custody import recorded_review_shape

    assert recorded_review_shape({"mode": "ask", "access": "readonly"}).access == "readonly"
    writable = {"mode": "agent", "access": "full",
                "execution": {"isolation": "live", "delegated": True, "workspaceRoot": str(tmp_path)}}
    assert recorded_review_shape(writable).investigation
    assert recorded_review_shape({"mode": "agent", "access": "full", "_use_thread": True}).investigation
    for invalid in ({"mode": "agent", "access": "full"}, {"mode": "ask", "access": "full"},
                    {"mode": "agent", "access": "readonly", "execution": writable["execution"]},
                    {"mode": "ask", "access": "readonly", "execution": {}}):
        assert recorded_review_shape(invalid) is None
    assert pathlib.Path(tmp_path).exists()


@pytest.mark.parametrize("surface", ["multi_model_review", "task_acceptance", "skill_review", "deep_self_review"])
@pytest.mark.parametrize("access", ["full", "readonly"])
def test_common_review_consumer_preserves_seat_access_and_evidence(
    repo, tmp_path, writable_engine, surface, access,
):
    from ouroboros.review_substrate import run_review_request
    from tests._review_session_route_shared import FakeLLM, _agent_request, _agent_slot

    drive = tmp_path / "drive"
    source = (review_checking.freeze_checking_source(repo, data_root=drive, subject=review_checking.SUBJECT_INDEX)
              if access == "full" else {})
    result = run_review_request(
        _agent_request(surface=surface, call_type=surface, task_id="task-1",
                       session_root=str(repo), checking_source=source),
        slots=[_agent_slot(slot_id="session", session_access=access)],
        drive_root=drive, llm=FakeLLM(),
    )
    body = FakeGateway.instances[-1].start_requests[0]
    assert body["access"] == access
    assert body["scope"]["root"] == str(repo)
    assert (body["mode"], "execution" in body) == (("agent", True) if access == "full" else ("ask", False))
    actor = result.actors[0]
    assert actor["usage"]["requested_access"] == access
    if access == "full":
        evidence = actor["usage"]["checking_evidence"]
        assert evidence["status"] == "retained"
        assert pathlib.Path(evidence["evidence_dir"], "summary.json").is_file()
        assert not pathlib.Path(body["execution"]["workspaceRoot"]).exists()
    else:
        assert "checking_evidence" not in actor["usage"]
    assert (repo / "f.txt").read_text() == "A staged\nB unstaged\n"
