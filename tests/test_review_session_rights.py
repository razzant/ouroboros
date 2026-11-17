"""Reviewer seats on the shared session transport: configured rights, private copies,
recorded-shape recovery and the raw-request replay a collection sends (no rebuilt prompt).
A copy whose own attempt failed before any start request is captured, then released.

Synthetic transport (``FakeGateway``): no live engine, no provider, no money.
"""
from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import review_checking, review_checking_prune
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.gateways.claudexor import ClaudexorGateway as _RealGateway, DaemonEndpoint as _Endpoint
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


def _raise(error):
    def fail(*_args, **_kwargs):
        raise error
    return fail


def _fail_before_send(monkeypatch, stage, area, meanwhile=lambda: None):
    """Fail the attempt at STAGE, after its copy exists and holds output, before any start request
    (``meanwhile`` first: another preparation of the same operation)."""
    from ouroboros import budget_pause, review_execution, review_session_preparation
    from ouroboros.usage_accounting import BillingAuthorityUnavailable

    def step(result=None, error=None):
        def run(*_args, **_kwargs):
            meanwhile()
            (area / "probe.txt").write_bytes(b"written before the failure\n")
            (area / review_checking.EVIDENCE_DIRNAME).mkdir(exist_ok=True)
            (area / review_checking.EVIDENCE_DIRNAME / "note.log").write_bytes(b"kept\n")
            if error is not None:
                raise error
            return result
        return run

    owner, name, effect = {  # inside the seat's preparation, then each later refusal before the start row
        "service": (review_session_preparation, "seat_service_directories", step(error=OSError("scratch grant unreadable"))),
        "request": (review_execution, "prepare_review_session_request",
                    step(error=ReviewRouteUnavailable("input too large", code="review_input_too_large"))),
        "budget": (budget_pause, "dispatch_fenced", step(result=True)),
        "deadline": (review_execution, "owner_deadline_exhausted", step(result=True)),
        "billing": (custody, "record_start_requested", step(error=BillingAuthorityUnavailable("no billing authority"))),
    }[stage]
    monkeypatch.setattr(owner, name, effect)


def _holds(drive):
    report = review_checking_prune.prune_review_checkouts(drive, set())
    assert "skipped" not in report, report
    return {row["class"] for row in report["holds"]}


@pytest.mark.parametrize("trust", ["timeout", "refused"])
def test_a_full_access_trust_failure_makes_no_checking_copy(repo, tmp_path, writable_engine, monkeypatch, trust):
    """The trust grant-or-refusal (a lookup that may POST a grant) runs before any copy exists, as for a thread's turn:
    a lookup that timed out (seat 006b7739…) or a refusal leaves no home, marker, copy or Git registration."""
    drive = tmp_path / "drive"
    source = review_checking.freeze_checking_source(repo, data_root=drive, subject=review_checking.SUBJECT_INDEX)
    failure = (ClaudexorUnavailable("daemon_timeout", "GET /v2/trust timed out") if trust == "timeout" else
               ClaudexorUnavailable("trust_full_access_required", "Full access is disabled", status_code=403))
    original = FakeGateway.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.ensure_full_access = lambda root: self.trust.append(root) or _raise(failure)()

    monkeypatch.setattr(FakeGateway, "__init__", init)
    stamps = []
    with pytest.raises(ClaudexorUnavailable) as caught:
        run_delegated_review_session(prompt="review", root=str(repo), custody_drive=drive, invocation=_invocation(
            access="full", checking_source=source, dispatch_stamp=lambda: stamps.append(1)))
    assert caught.value is failure and caught.value.delegated_start_unrequested is True
    gateway = FakeGateway.instances[-1]
    assert gateway.trust == [str(repo)] and not gateway.start_requests and not stamps
    assert not review_checking.seat_dir(drive, "task-1", "op-1").exists()
    assert _git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1


@pytest.mark.parametrize("stage", ["service", "request", "budget", "deadline", "billing"])
def test_a_failure_before_any_start_request_captures_the_new_copy_then_releases_it(
        repo, tmp_path, writable_engine, monkeypatch, stage):
    """The attempt that made the copy failed before any start request: the receipt of what the copy holds precedes
    its release, through the terminal capture; outer facts survive a failure inside the seat's own preparation."""
    drive = tmp_path / "drive"
    source = review_checking.freeze_checking_source(repo, data_root=drive, subject=review_checking.SUBJECT_INDEX)
    area = review_checking.seat_dir(drive, "task-1", "op-1") / "area.noindex"
    _fail_before_send(monkeypatch, stage, area)
    stamps = []
    with pytest.raises(Exception) as caught:
        run_delegated_review_session(prompt="review", root=str(repo), custody_drive=drive, invocation=_invocation(
            access="workspace_write", checking_source=source, dispatch_stamp=lambda: stamps.append(1),
            owner_deadline_at="2099-01-01T00:00:00+00:00" if stage == "deadline" else ""))
    receipt = caught.value.checking_evidence
    assert caught.value.delegated_start_unrequested is True and not stamps
    assert not FakeGateway.instances[-1].start_requests
    assert receipt["status"] == "retained" and "A\tprobe.txt" in receipt["changed"], receipt
    evidence = pathlib.Path(receipt["evidence_dir"])
    assert json.loads((evidence / "summary.json").read_text(encoding="utf-8"))["changed"] == receipt["changed"]
    assert (evidence / "files" / review_checking.EVIDENCE_DIRNAME / "note.log").read_bytes() == b"kept\n"
    assert not area.exists() and _git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1
    assert (repo / "f.txt").read_text() == "A staged\nB unstaged\n" and _git(repo, "write-tree") == source["tree"]
    if stage != "service":  # a minted invocation's refusal row makes the custody store answerable
        assert "capture_unproven" not in _holds(drive)


@pytest.mark.parametrize("case", ["unknown_send", "ambiguous_row", "predecessor", "concurrent", "keep", "capture_failed"])
def test_a_copy_not_proven_unused_by_the_attempt_that_made_it_stays(repo, tmp_path, writable_engine, monkeypatch, case):
    """An unknown send or start row, a copy reused from a predecessor or by a concurrent preparation of the same
    operation, the owner's Keep and a failed capture keep the copy; an unknown send still replays into it."""
    from ouroboros import owner_keep
    from tests._store_shared import seed_result

    drive = tmp_path / "drive"
    source = review_checking.freeze_checking_source(repo, data_root=drive, subject=review_checking.SUBJECT_INDEX)
    area = review_checking.seat_dir(drive, "task-1", "op-1") / "area.noindex"

    def same_operation():  # under the seat lock: the copy is reused as it is
        assert review_checking.prepare_seat_area(drive, "task-1", "op-1", source) == str(area)

    if case == "unknown_send":
        FakeGateway.start_error = TimeoutError("accepted response lost")
    elif case == "ambiguous_row":
        monkeypatch.setattr(custody, "record_start_requested", _raise(OSError("start row outcome unknown")))
    else:
        if case == "predecessor":
            same_operation()
            (area / "earlier.txt").write_bytes(b"a predecessor's work\n")
        elif case == "keep":
            seed_result(drive, "task-1", {"status": "running"})
            assert owner_keep.set_owner_keep(drive, "task-1", keep=True, source={"basis": "test"})["keep"] is True
        elif case == "capture_failed":
            write = review_checking.atomic_write_json
            monkeypatch.setattr(review_checking, "atomic_write_json", lambda path, *a, **k: _raise(OSError(
                "evidence volume unavailable"))() if pathlib.Path(path).name == "summary.json" else write(path, *a, **k))
        _fail_before_send(monkeypatch, "request", area, meanwhile=same_operation if case == "concurrent" else lambda: None)
    state = {}
    with pytest.raises(Exception) as caught:
        run_delegated_review_session(prompt="review", root=str(repo), custody_drive=drive, invocation=_invocation(
            access="workspace_write", checking_source=source, retry_state=state))
    receipt = getattr(caught.value, "checking_evidence", {})
    assert area.is_dir() and (receipt.get("status") or "") == {"keep": "retained", "capture_failed": "capture_failed"}.get(case, "")
    assert (getattr(caught.value, "delegated_start_unrequested", None) is True) == (case not in ("unknown_send", "ambiguous_row"))
    if case == "predecessor":
        assert (area / "earlier.txt").read_bytes() == b"a predecessor's work\n"
    # Without a durable receipt (a failed summary write too) the copy keeps the hold it had before this repair.
    assert {"unknown_send": "run_open", "keep": owner_keep.OWNER_KEEP}.get(case, "capture_unproven") in _holds(drive)
    if case == "unknown_send":  # the explicit retry replays its exact start into the copy it kept
        token = state["pending_invocation_id"]
        facts = run_delegated_review_session(prompt="review", root=str(repo), custody_drive=drive, invocation=_invocation(
            access="workspace_write", checking_source=source, retry_state=state))
        assert FakeGateway.instances[-1].start_keys == [token]
        assert facts["checking_evidence"]["status"] == "retained" and not area.exists()


def _fenced_before_post(monkeypatch, meanwhile=lambda: None):
    """The launch fence refuses this attempt's recorded start before its POST is entered."""
    from ouroboros import review_execution
    from ouroboros.owner_pause import OwnerPauseRefused

    def fenced(*_args, **_kwargs):
        meanwhile()
        raise OwnerPauseRefused("owner_pause_fenced")

    monkeypatch.setattr(review_execution, "run_operation", fenced)


def _engine_refusal(code, status, problem=None):
    error = ClaudexorUnavailable(code, "the engine refused the start", status_code=status)
    error.problem = problem or {"code": code, "message": "the engine refused the start"}  # the received ControlProblem
    return error


def _synthesized_refusal(status):
    """What the real gateway raises for an engine answer whose body is not JSON: a typed-looking ControlProblem it
    synthesizes itself (``http_<status>``), indistinguishable by shape from an engine's own."""
    import httpx

    gateway = _RealGateway(_Endpoint(host="127.0.0.1", port=1, token="fixture-token"))  # bound before fixtures patch it
    try:
        error = gateway._problem(httpx.Response(status, text="<html>proxy refused the request</html>"), "POST")
    finally:
        gateway.close()
    assert (error.code, error.problem["code"]) == (f"http_{status}", f"http_{status}")
    return error


# Every answer an ENTERED POST can get. None proves the start created nothing (#3138, audit 980c602f): a body may
# be synthesized, and an accepted command can end before execution and answer its own start with a 4xx.
ANSWERED_REFUSALS = {
    "engine_4xx": lambda: _engine_refusal("request_invalid", 400),
    "accepted_then_400": lambda: _engine_refusal("execution_workspace_invalid", 400, {
        "code": "execution_workspace_invalid", "message": "the job's workspace is gone",
        "context": {"jobId": "job-7", "state": "failed"}}),  # the audit's accepted-then-terminal answer
    "non_json_4xx": lambda: _synthesized_refusal(400),
    "handshake_426": lambda: _engine_refusal("handshake_required", 426),
    "retired_profile_409": lambda: _engine_refusal("retired_access_profile", 409),
    "idempotency_conflict": lambda: _engine_refusal("idempotency_conflict", 409),
    "transport_4xx": lambda: ClaudexorUnavailable("daemon_unreachable", "response body lost", status_code=400),
}


def _invocation_ids(drive, kind):
    return [row["invocation_id"] for row in custody.custody_rows(drive) if row.get("type") == kind]


@pytest.mark.parametrize("stage", ["launch_fence", "paid_stamp", "row_landed_reported_false"])
def test_a_recorded_start_that_provably_never_ran_releases_its_new_copy(repo, tmp_path, writable_engine, monkeypatch, stage):
    """After its START_REQUESTED write the attempt never entered its POST (the launch fence, a failed paid stamp, a
    row that landed although its write reported otherwise), and the invocation reads back ``failed_definite``
    without a run for this seat and copy: receipt, then release. An entered POST never releases (below)."""
    from ouroboros import review_execution

    drive = tmp_path / "drive"
    source = review_checking.freeze_checking_source(repo, data_root=drive, subject=review_checking.SUBJECT_INDEX)
    area = review_checking.seat_dir(drive, "task-1", "op-1") / "area.noindex"
    if stage == "launch_fence":
        _fenced_before_post(monkeypatch)
    elif stage == "paid_stamp":
        monkeypatch.setattr(review_execution, "invoke_review_paid_stamp", _raise(OSError("paid stamp outcome unknown")))
    else:
        record = custody.record_start_requested
        monkeypatch.setattr(custody, "record_start_requested", lambda *a, **k: record(*a, **k) and False)
    state = {}
    with pytest.raises(Exception) as caught:
        run_delegated_review_session(prompt="review", root=str(repo), custody_drive=drive, invocation=_invocation(
            access="workspace_write", checking_source=source, retry_state=state))
    [token] = _invocation_ids(drive, custody.START_REQUESTED)
    assert custody.invocation_record(drive, token)["state"] == "failed_definite" and "pending_invocation_id" not in state
    assert getattr(caught.value, "delegated_start_unrequested", None) is not True  # its start row was written
    assert FakeGateway.instances[-1].start_requests == []
    receipt = caught.value.checking_evidence
    assert receipt["status"] == "retained" and pathlib.Path(receipt["evidence_dir"], "summary.json").is_file(), receipt
    assert not area.exists() and _git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1
    assert "capture_unproven" not in _holds(drive)


@pytest.mark.parametrize("case", ["lost_write", "failed_emit", "late_started", "legacy_gap", "unreadable_fate",
                                  "reused", *ANSWERED_REFUSALS])
def test_a_recorded_start_not_proven_unrun_keeps_its_copy(repo, tmp_path, writable_engine, monkeypatch, case):
    """A start row that never landed (no fate), a refusal row that never landed (still pending, replayable), a
    STARTED bound before the read (late STARTED keeps precedence), a legacy or unreadable fate, a copy another
    preparation of the same operation reused, and every answer to an entered POST (``ANSWERED_REFUSALS``: a typed
    4xx, the audit's accepted-then-400, a synthesized non-JSON body, 426/409 refusals, ``idempotency_conflict``,
    a transport failure carrying a 4xx status): nothing proves this attempt's start never ran in this copy, so the
    copy stays. An answered start also keeps its pending token and its fate unknown, never ``failed_definite``;
    the engine here declares no accepted-start read, so nothing is bound and nothing is sent again."""
    from ouroboros.store import StoreError

    drive = tmp_path / "drive"
    source = review_checking.freeze_checking_source(repo, data_root=drive, subject=review_checking.SUBJECT_INDEX)
    area = review_checking.seat_dir(drive, "task-1", "op-1") / "area.noindex"
    emit, fate, overriding = custody.emit, custody.invocation_record, [True]

    def started_first(drive_root, kind, payload, **kwargs):  # a run bound to this start before its refusal lands
        if kind == custody.START_FAILED:
            entry = custody.RunCustody(run_id="run-late", task_id="task-1", route_id="fake-review",
                                       invocation_id=payload["invocation_id"], idempotency_key=payload["invocation_id"],
                                       ledger_root=str(drive_root), authority_source="investigation",
                                       execution_root=str(area), target_root=str(repo))
            assert custody.bind_started(drive_root, entry, shape={"access": "workspace_write", "mode": "agent"})[1]
        return emit(drive_root, kind, payload, **kwargs)

    def unproven_fate(drive_root, token, **kwargs):
        if overriding[0] and case == "unreadable_fate":
            raise StoreError("custody store unreadable")
        return {"state": "legacy_gap", "run_id": "", "reason": "fixture"} if overriding[0] else fate(drive_root, token, **kwargs)

    answered = case in ANSWERED_REFUSALS
    if case == "lost_write":
        monkeypatch.setattr(custody, "record_start_requested", lambda *a, **k: False)
    elif answered:
        FakeGateway.start_error = ANSWERED_REFUSALS[case]()
    else:
        _fenced_before_post(monkeypatch, meanwhile=(lambda: review_checking.prepare_seat_area(
            drive, "task-1", "op-1", source)) if case == "reused" else lambda: None)
        if case in ("failed_emit", "late_started"):
            monkeypatch.setattr(custody, "emit", started_first if case == "late_started" else (
                lambda drive_root, kind, payload, **k: False if kind == custody.START_FAILED else emit(drive_root, kind, payload, **k)))
        elif case in ("legacy_gap", "unreadable_fate"):
            monkeypatch.setattr(custody, "invocation_record", unproven_fate)
    state = {}
    with pytest.raises(Exception) as caught:
        run_delegated_review_session(prompt="review", root=str(repo), custody_drive=drive, invocation=_invocation(
            access="workspace_write", checking_source=source, retry_state=state))
    overriding[0] = False
    token = (_invocation_ids(drive, custody.START_REQUESTED) or _invocation_ids(drive, custody.START_FAILED))[0]
    record = custody.invocation_record(drive, token)
    assert (record or {}).get("state") == {"lost_write": None, "late_started": "started"}.get(
        case, "pending" if answered or case == "failed_emit" else "failed_definite"), record  # failed_definite alone proves nothing
    assert area.is_dir() and not getattr(caught.value, "checking_evidence", None)
    assert not list(area.parent.glob("evidence*/summary.json"))
    gateway = FakeGateway.instances[-1]
    assert (len(gateway.start_requests), gateway.removals) == (1 if answered else 0, [])  # no second POST, nothing retired
    assert (state.get("pending_invocation_id") == token) is answered  # an answered start keeps its token
    if answered:
        assert caught.value is not None and caught.value.accepted_start_lookup["lookup"] == "unsupported"
        failed = [row for row in custody.custody_rows(drive) if row.get("type") == custody.START_FAILED]
        assert [(row["definite"], row["project_retention_reason"]) for row in failed] == [
            (False, "start_outcome_unknown_run_may_exist")]
    assert ("run_open" if answered or case in ("failed_emit", "late_started") else "capture_unproven") in _holds(drive)


def test_a_recovery_fenced_before_its_own_post_keeps_the_copy_and_the_token(repo, tmp_path, writable_engine,
                                                                            monkeypatch):
    """A recovery that never entered its own POST proves nothing about the attempt it recovers: the first POST's
    outcome stays unknown, so its copy (with whatever ran there), its token and its pending fate all stay."""
    drive = tmp_path / "drive"
    source = review_checking.freeze_checking_source(repo, data_root=drive, subject=review_checking.SUBJECT_INDEX)
    area = review_checking.seat_dir(drive, "task-1", "op-1") / "area.noindex"
    FakeGateway.start_error = TimeoutError("accepted response lost")  # the first POST's outcome is unknown
    state = {}
    with pytest.raises(TimeoutError):
        run_delegated_review_session(prompt="review", root=str(repo), custody_drive=drive, invocation=_invocation(
            access="workspace_write", checking_source=source, retry_state=state))
    token = state["pending_invocation_id"]
    (area / "unknown-run.txt").write_bytes(b"what the unknown run may have written\n")
    _fenced_before_post(monkeypatch)
    with pytest.raises(ReviewRouteUnavailable) as caught:
        run_delegated_review_session(prompt="review", root=str(repo), custody_drive=drive, invocation=_invocation(
            access="workspace_write", checking_source=source, retry_state=state))
    assert caught.value.code == "owner_pause_fenced" and state["pending_invocation_id"] == token
    assert custody.invocation_record(drive, token)["state"] == "pending"
    assert [row["definite"] for row in custody.custody_rows(drive)
            if row.get("type") == custody.START_FAILED and row.get("invocation_id") == token] == [False, False]
    assert (area / "unknown-run.txt").read_bytes() == b"what the unknown run may have written\n"
    assert not getattr(caught.value, "checking_evidence", None) and not list(area.parent.glob("evidence*/summary.json"))
    assert [request for gateway in FakeGateway.instances for request in gateway.start_requests] == [
        FakeGateway.instances[0].start_requests[0]]  # only the original POST: the recovery sent nothing
    assert "run_open" in _holds(drive)


_RUN_LOOKUP_CATALOG = [{"method": "POST", "path": "/v2/runs/lookup"}]  # the engine's declared noncreating read


def _engine_reads(monkeypatch, answer):
    """The engine declares ``POST /v2/runs/lookup`` and answers it with ``answer`` (a body, or an error raised)."""
    import copy

    reads = []

    def lookup_run(self, request, *, idempotency_key):
        reads.append((idempotency_key, dict(request)))
        if isinstance(answer, BaseException):
            raise answer
        return copy.deepcopy(answer)
    monkeypatch.setattr(FakeGateway, "operations", lambda self: _RUN_LOOKUP_CATALOG, raising=False)
    monkeypatch.setattr(FakeGateway, "lookup_run", lookup_run, raising=False)
    return reads


WINDOW_FAILURE = {"code": "subscription_window_exhausted", "resetsAt": "2026-11-16T12:00:00Z",
                  "safeMessage": "Subscription window spent before the run started"}


@pytest.mark.parametrize("timing", ["early_4xx", "late_202"])
def test_an_accepted_job_answered_early_or_late_has_one_fate(repo, tmp_path, writable_engine, monkeypatch, timing):
    """The engine accepts the start as job-7, which ends before any run. Within its bind wait the POST answers with
    the job's own 4xx (``early_4xx``, the audit's case), later with 202 and the job id (``late_202``). Timing must
    not decide the copy: the noncreating read binds the job once, and the ordinary terminal path captures, then
    releases the copy and settles once, reporting the job's own typed failure and reset, never a success."""
    from ouroboros.gateways.claudexor import ClaudexorSubscriptionWindowExhausted
    from tests._review_session_route_shared import _terminal_detail

    drive = tmp_path / "drive"
    source = review_checking.freeze_checking_source(repo, data_root=drive, subject=review_checking.SUBJECT_INDEX)
    area = review_checking.seat_dir(drive, "task-1", "op-1") / "area.noindex"
    detail = _terminal_detail("", state="failed")
    detail["summary"].update(jobId="job-7", runId=None, failure=dict(WINDOW_FAILURE))
    for unreported in ("spendUsd", "spendEstimated"):
        detail["summary"].pop(unreported)  # no cost reported: unknown, never zero
    FakeGateway.detail = detail
    reads = _engine_reads(monkeypatch, {"status": "found", "storeId": "store-a", "handle": {
        "jobId": "job-7", "runId": None, "state": "failed", "runDir": None, "taskId": None,
        "problem": {"code": "subscription_window_exhausted", "message": "spent"}}})

    def start_run(self, request, *, idempotency_key=""):
        self.start_requests.append(dict(request))
        self.start_keys.append(str(idempotency_key))
        if timing == "early_4xx":
            raise _engine_refusal("subscription_window_exhausted", 429, {
                "code": "subscription_window_exhausted", "message": "spent", "context": {"jobId": "job-7", "state": "failed"}})
        return {"jobId": "job-7", "runId": None}
    monkeypatch.setattr(FakeGateway, "start_run", start_run)
    state = {}
    with pytest.raises(ClaudexorSubscriptionWindowExhausted) as caught:
        run_delegated_review_session(prompt="review", root=str(repo), custody_drive=drive, invocation=_invocation(
            access="workspace_write", checking_source=source, retry_state=state))
    failure = caught.value
    assert (failure.code, failure.reset_at, failure.delegated_run_id) == (
        "subscription_window_exhausted", "2026-11-16T12:00:00Z", "job-7"), "the job's own typed stop, kept"
    [token] = FakeGateway.instances[-1].start_keys  # one POST, never a second
    assert [key for key, _body in reads] == ([token] if timing == "early_4xx" else [])
    record = custody.invocation_record(drive, token)
    assert (record["state"], record["run_id"]) == ("started", "job-7") and "pending_invocation_id" not in state
    rows = custody.custody_rows(drive)
    started = [row for row in rows if row.get("type") == custody.STARTED and row.get("invocation_id") == token]
    settled = [row for row in rows if row.get("type") == custody.SETTLED and row.get("run_id") == "job-7"]
    assert len(started) == 1 and [(row["state"], row["cost_usd"]) for row in settled] == [("failed", None)]
    assert not [row for row in rows if row.get("type") == custody.START_FAILED]
    if timing == "early_4xx":  # the read's binding names what it read behind
        assert started[0]["accepted_start_lookup"] == "found"
        assert started[0]["start_refusal"] == {"code": "subscription_window_exhausted", "status": 429}
    receipt = failure.checking_evidence
    assert receipt["status"] == "retained" and pathlib.Path(receipt["evidence_dir"], "summary.json").is_file(), receipt
    assert not area.exists() and FakeGateway.instances[-1].removals == []
    assert not {"run_open", "capture_unproven"} & _holds(drive)


@pytest.mark.parametrize("answer", ["not_found", "store_mismatch", "refused", "unavailable", "late_started"])
def test_an_answered_start_whose_read_binds_nothing_keeps_all_and_sends_nothing(repo, tmp_path, writable_engine,
                                                                                monkeypatch, answer):
    """The POST was answered 400; the engine's read binds nothing: its store holds no binding (``not_found``, no
    proof the POST created nothing), another store answers, the read is refused or unavailable, or another reader
    already bound the start (``late_started``). Nothing is created (no second POST, no implicit create on a
    miss), nothing is removed (copy, registration) and the token stays: the outcome stays unknown."""
    drive = tmp_path / "drive"
    source = review_checking.freeze_checking_source(repo, data_root=drive, subject=review_checking.SUBJECT_INDEX)
    area = review_checking.seat_dir(drive, "task-1", "op-1") / "area.noindex"
    found_elsewhere = {"status": "found", "storeId": "store-b", "handle": {
        "jobId": "job-9", "runId": None, "state": "failed", "runDir": None, "taskId": None}}
    reads = _engine_reads(monkeypatch, {
        "not_found": {"status": "not_found", "storeId": "store-a"}, "store_mismatch": found_elsewhere,
        "refused": _engine_refusal("idempotency_conflict", 409),
        "unavailable": ClaudexorUnavailable("daemon_busy", "busy", status_code=503),
    }.get(answer, found_elsewhere))
    if answer == "store_mismatch":
        monkeypatch.setattr(FakeGateway, "engine_ref", {"store": "store-a"}, raising=False)  # the store it proved
    refusal = _engine_refusal("execution_workspace_invalid", 400)

    def start_run(self, request, *, idempotency_key=""):
        self.start_requests.append(dict(request))
        self.start_keys.append(str(idempotency_key))
        if answer == "late_started":  # another reader binds this very start before its answer is handled
            entry = custody.RunCustody(run_id="run-late", task_id="task-1", route_id="fake-review",
                                       invocation_id=str(idempotency_key), idempotency_key=str(idempotency_key),
                                       ledger_root=str(drive), authority_source="investigation",
                                       execution_root=str(area), target_root=str(repo))
            assert custody.bind_started(drive, entry, shape={"access": "workspace_write", "mode": "agent"})[1]
        raise refusal
    monkeypatch.setattr(FakeGateway, "start_run", start_run)
    state = {}
    with pytest.raises(ClaudexorUnavailable) as caught:
        run_delegated_review_session(prompt="review", root=str(repo), custody_drive=drive, invocation=_invocation(
            access="workspace_write", checking_source=source, retry_state=state))
    assert caught.value is refusal and state["pending_invocation_id"] == FakeGateway.instances[-1].start_keys[0]
    token = state["pending_invocation_id"]
    gateway = FakeGateway.instances[-1]
    assert (len(gateway.start_requests), gateway.removals) == (1, []) and len(reads) == (answer != "late_started")
    expected = {"not_found": "not_found", "store_mismatch": "store_mismatch", "refused": "refused",
                "unavailable": "unavailable"}.get(answer)
    assert caught.value.accepted_start_lookup.get("lookup", "not_read") == (expected or "not_read")
    record = custody.invocation_record(drive, token)
    assert (record["state"], record["run_id"]) == (("started", "run-late") if answer == "late_started" else ("pending", ""))
    rows = custody.custody_rows(drive)
    assert len([row for row in rows if row.get("type") == custody.STARTED]) == (answer == "late_started")
    assert [row["definite"] for row in rows if row.get("type") == custody.START_FAILED] == [False]
    assert area.is_dir() and not getattr(caught.value, "checking_evidence", None)
    assert "run_open" in _holds(drive)


@pytest.mark.parametrize("stage", ["before_the_row", "launch_fence"])
def test_a_failed_release_never_replaces_the_attempts_own_refusal(repo, tmp_path, writable_engine, monkeypatch, stage):
    """An unreadable install identity at the release (``owner_keep.gate`` raises ``StoreError``, not an OSError) after
    the receipt is written: the attempt's own typed refusal stays its outcome, the evidence names the failure and the
    copy stays with its receipt for the prune."""
    import contextlib

    from ouroboros import budget_pause, owner_keep
    from ouroboros.store import StoreError

    drive = tmp_path / "drive"
    source = review_checking.freeze_checking_source(repo, data_root=drive, subject=review_checking.SUBJECT_INDEX)
    area = review_checking.seat_dir(drive, "task-1", "op-1") / "area.noindex"
    if stage == "before_the_row":
        monkeypatch.setattr(budget_pause, "dispatch_fenced", lambda *_a, **_k: True)
    else:
        _fenced_before_post(monkeypatch)

    @contextlib.contextmanager
    def unreadable_identity(*_args, **_kwargs):
        raise StoreError("install identity unreadable")
        yield  # pragma: no cover - a generator, never reached

    monkeypatch.setattr(owner_keep, "gate", unreadable_identity)
    with pytest.raises(ReviewRouteUnavailable) as caught:
        run_delegated_review_session(prompt="review", root=str(repo), custody_drive=drive, invocation=_invocation(
            access="workspace_write", checking_source=source))
    assert caught.value.code == ("budget_pausing_no_send" if stage == "before_the_row" else "owner_pause_fenced")
    assert (getattr(caught.value, "delegated_start_unrequested", None) is True) == (stage == "before_the_row")
    evidence = caught.value.checking_evidence
    assert evidence["status"] == "capture_failed" and "StoreError: install identity unreadable" in evidence["error"]
    assert area.is_dir() and list(area.parent.glob("evidence*/summary.json"))  # the receipt preceded the failed release


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


@pytest.mark.parametrize(("access", "surface"), [
    ("full", "multi_model_review"),
    ("full", "deep_self_review"),
    ("readonly", "task_acceptance"),
    ("readonly", "skill_review"),
])
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


@pytest.mark.parametrize("started", [False, True])
def test_a_legacy_writable_successor_read_uses_the_original_copy_and_captures_its_output(
        repo, tmp_path, writable_engine, monkeypatch, started):
    """A legacy successor's lookup binding rejoins the original checking allocation; successful terminal
    collection captures its bytes, releases the area and leaves exactly one cost row per actual run."""
    from tests.test_review_session_continuation import _engine, record_legacy_successor

    drive = tmp_path / "drive"
    source = review_checking.freeze_checking_source(repo, data_root=drive, subject=review_checking.SUBJECT_INDEX)
    posts = _engine(monkeypatch, stops=1)
    checkpoints = []
    def run(state):
        return run_delegated_review_session(prompt="review", root=str(repo), custody_drive=drive,
            invocation=_invocation(access="full", checking_source=source, retry_state=state,
                                   pending_invocation_checkpoint=checkpoints.append))
    with pytest.raises(ClaudexorUnavailable) as failure:
        run({})
    assert failure.value.code == "credential_pool_exhausted"
    area = review_checking.seat_dir(drive, "task-1", "op-1") / "area.noindex"
    assert area.is_dir()
    # Exact bytes on every host: capture hashes raw content (no eol conversion), so text mode's CRLF would be kept.
    (area / "successor-output.txt").write_bytes(b"unique successor output\n")
    token = record_legacy_successor(drive, checkpoints[0], started=started)
    reads = []
    monkeypatch.setattr(FakeGateway, "operations", lambda _s: [{"method": "POST", "path": "/v2/runs/lookup"}],
                        raising=False)
    def lookup(_self, request, *, idempotency_key=""):
        reads.append((request, idempotency_key))
        return {"status": "found", "storeId": "store-r", "handle": {
            "jobId": "job-2", "runId": "run-2", "state": "succeeded"}}
    monkeypatch.setattr(FakeGateway, "lookup_run", lookup, raising=False)
    facts = run({"pending_invocation_id": checkpoints[0]})
    assert facts["run_id"] == "run-2" and facts["text"] == "[]"
    assert reads == ([] if started else [({"continueFrom": "run-1"}, token)]) and len(posts) == 1
    assert facts["checking_evidence"]["status"] == "retained" and not area.exists()
    evidence = pathlib.Path(facts["checking_evidence"]["evidence_dir"])
    assert (evidence / "summary.json").is_file()
    assert b"+unique successor output\n" in (evidence / "delta.patch").read_bytes()
    rows = list(custody.custody_rows(drive))
    assert sorted(row["run_id"] for row in rows if row["type"] == custody.LEDGER_RECORDED) == ["run-1", "run-2"]
    assert custody.replay(drive)["run-2"].execution_root == str(area)


@pytest.mark.parametrize("gap", ["missing", "foreign"])
@pytest.mark.parametrize("started", [False, True])
def test_a_bound_legacy_successor_without_its_copy_is_observed_with_a_typed_capture_gap(
        repo, tmp_path, writable_engine, monkeypatch, started, gap):
    """The successor bound under this seat's own legacy token is collected and settled even when its checking copy
    is gone or now held by another operation: its answer and run reach the seat, its evidence is a typed gap, no
    copy is made, taken or released, the seat's earlier capture stays byte-identical and nothing is re-POSTed."""
    from tests.test_review_session_continuation import _engine, record_legacy_successor

    drive = tmp_path / "drive"
    source = review_checking.freeze_checking_source(repo, data_root=drive, subject=review_checking.SUBJECT_INDEX)
    posts = _engine(monkeypatch, stops=1)
    checkpoints = []
    def run(state):
        return run_delegated_review_session(prompt="review", root=str(repo), custody_drive=drive,
            invocation=_invocation(access="full", checking_source=source, retry_state=state,
                                   pending_invocation_checkpoint=checkpoints.append))
    with pytest.raises(ClaudexorUnavailable):
        run({})
    home = review_checking.seat_dir(drive, "task-1", "op-1")
    area, marker = home / "area.noindex", home / "checking.json"
    assert area.is_dir() and list(home.glob("evidence*")), "the stopped predecessor's capture exists"
    if gap == "missing":
        area.rename(home / "moved-away")
    else:
        marker.write_text(json.dumps({**json.loads(marker.read_text()), "holder_operation_id": "newer-operation"}))
    def files():
        return {p.relative_to(home): p.read_bytes() for p in home.rglob("*") if p.is_file() and p.name != ".lock"}
    before = files()
    token = record_legacy_successor(drive, checkpoints[0], started=started)
    def lookup(_self, request, *, idempotency_key=""):
        return {"status": "found", "storeId": "store-r", "handle": {
            "jobId": "job-2", "runId": "run-2", "state": "succeeded"}}
    monkeypatch.setattr(FakeGateway, "lookup_run", lookup, raising=False)
    facts = run({"pending_invocation_id": checkpoints[0]})
    assert facts["run_id"] == "run-2" and facts["text"] == "[]" and len(posts) == 1
    assert [step["run_id"] for step in facts["review_continuations"]] == ["run-2"]
    evidence = facts["checking_evidence"]
    assert evidence["status"] == "capture_failed" and evidence["run_id"] == "run-2"
    assert evidence["capture_gap"] == "legacy_successor_checking_copy_unavailable"
    assert "CheckingAreaUnavailable" in evidence["error"]
    assert files() == before, "no copy restored, taken or released; earlier captures and the marker unchanged"
    assert area.exists() == (gap == "foreign")
    rows = list(custody.custody_rows(drive))
    assert sorted(row["run_id"] for row in rows if row["type"] == custody.LEDGER_RECORDED) == ["run-1", "run-2"]
    assert custody.invocation_record(drive, token)["state"] == "started"


@pytest.mark.parametrize("gap", ["missing", "foreign"])
@pytest.mark.parametrize("started", [False, True])
def test_a_lost_successor_poll_keeps_the_typed_capture_gap_on_the_pending_error_actor(
        repo, tmp_path, writable_engine, monkeypatch, started, gap):
    """The bound legacy successor without its copy, whose poll then answers 404: the start stays pending (no settled
    failure, nothing re-POSTed) and the typed capture gap reaches the common consumer's error actor and its durable
    error response, not only the session runner's exception."""
    from ouroboros.observability import read_call_payload
    from ouroboros.review_session_preparation import render_review_session_prompt
    from ouroboros.review_substrate import ReviewCoordinator
    from tests._review_session_route_shared import FakeLLM, _agent_request, _agent_slot
    from tests.test_review_session_continuation import _engine, record_legacy_successor

    drive = tmp_path / "drive"
    source = review_checking.freeze_checking_source(repo, data_root=drive, subject=review_checking.SUBJECT_INDEX)
    request = _agent_request(surface="multi_model_review", call_type="multi_model_review", task_id="task-1",
                             session_root=str(repo), checking_source=source)
    slot = _agent_slot(slot_id="session", session_access="full")
    posts = _engine(monkeypatch, stops=1)
    checkpoints = []
    with pytest.raises(ClaudexorUnavailable):  # the stopped predecessor, under the seat's exact prompt
        run_delegated_review_session(prompt=render_review_session_prompt(request, slot, request.session_task),
            root=str(repo), custody_drive=drive, invocation=_invocation(slot_id="session", access="full",
                checking_source=source, pending_invocation_checkpoint=checkpoints.append))
    home = review_checking.seat_dir(drive, "task-1", "op-1")
    area, marker = home / "area.noindex", home / "checking.json"
    if gap == "missing":
        area.rename(home / "moved-away")
    else:
        marker.write_text(json.dumps({**json.loads(marker.read_text()), "holder_operation_id": "newer-operation"}))
    def files():
        return {p.relative_to(home): p.read_bytes() for p in home.rglob("*") if p.is_file() and p.name != ".lock"}
    before = files()
    record_legacy_successor(drive, checkpoints[0], started=started)
    monkeypatch.setattr(FakeGateway, "lookup_run", lambda _s, _request, **_k: {"status": "found", "storeId": "store-r",
        "handle": {"jobId": "job-2", "runId": "run-2", "state": "succeeded"}}, raising=False)
    engine_detail = FakeGateway.get_run
    def successor_gone(self, run_id, **kwargs):
        if run_id != "run-2":
            return engine_detail(self, run_id, **kwargs)
        self.run_gets.append(run_id)
        raise ClaudexorUnavailable("run_not_found", "run run-2 is not known to this engine", status_code=404)
    monkeypatch.setattr(FakeGateway, "get_run", successor_gone)
    actor = ReviewCoordinator(llm=FakeLLM(), drive_root=drive)._run_slot(
        request, slot, operation_id="op-1", retry_state={"pending_invocation_id": checkpoints[0]})
    assert actor.status == "error" and any("run-2" in gateway.run_gets for gateway in FakeGateway.instances)
    assert len(posts) == 1 and actor.late_result_pending
    assert actor.recovery_binding["pending_invocation_id"] == checkpoints[0]
    assert actor.usage["pending_invocation_id"] == checkpoints[0] and actor.usage["delegated_run_id"] == "run-2"
    evidence = actor.usage["checking_evidence"]
    assert evidence["status"] == "capture_failed" and evidence["run_id"] == "run-2"
    assert evidence["capture_gap"] == "legacy_successor_checking_copy_unavailable"
    assert "CheckingAreaUnavailable" in evidence["error"]
    durable = read_call_payload(drive, task_id="task-1", call_id="op-1_error")[1]["usage"]["checking_evidence"]
    assert (durable["capture_gap"], durable["run_id"]) == (evidence["capture_gap"], "run-2")
    assert files() == before, "no copy restored, taken or released; earlier captures and the marker unchanged"


_RETAINED = {"status": "retained", "evidence_dir": "/evidence/op-1"}
_GAP = {"status": "capture_failed", "capture_gap": "legacy_successor_checking_copy_unavailable", "run_id": "run-2"}
_COLLECTED = {"run_id": "run-3", "text": "[]", "conformance": "passed", "schema_asked": True, "settlement": "settled",
              "route_id": "fake-review", "effective_route_ids": ["fake-review"], "model": "api/model-a",
              "spend": 0.0, "spend_estimated": False}


def _seat(tmp_path, monkeypatch, attempts):
    """A real session seat whose shared runner answers each attempt in turn with ``(outcome, pending)``: the failure it
    raises or the facts of a settled collection, and whether the start token stays pending, as the runner leaves it."""
    from ouroboros import review_execution as rx
    from tests._review_session_route_shared import _agent_request, _agent_slot

    executor = rx.AgentSessionReviewExecutor(rx.ReviewAssignment(
        _agent_request(session_root=str(tmp_path), task_id="task-1"), _agent_slot(), custody_root=tmp_path,
        call_id="op-1"))
    executor.usage_observer = [].append
    queue = list(attempts)
    def run(*, invocation, **_kwargs):
        outcome, pending = queue.pop(0)
        if pending:
            invocation.retry_state["pending_invocation_id"] = "inv-1"
        else:
            invocation.retry_state.pop("pending_invocation_id", None)
        if isinstance(outcome, BaseException):
            raise outcome
        return dict(outcome)
    monkeypatch.setattr(rx, "run_delegated_review_session", run)
    return executor


def _poll_failure(evidence=None):
    failure = ClaudexorUnavailable("run_not_found", "run run-2 is not known to this engine", status_code=404)
    failure.delegated_run_started, failure.delegated_run_id = True, "run-2"
    if evidence is not None:
        failure.checking_evidence = evidence
    return failure


@pytest.mark.parametrize(("pending", "evidence", "carried"), [
    (False, _RETAINED, True),  # healthy settled receipt: carried before and after
    (True, _RETAINED, True),  # healthy receipt of a pending start
    (True, _GAP, True),  # the typed capture gap of a pending start
    (True, None, False),  # absent: no receipt is invented
    (True, {}, False),
    (True, "capture_failed", False),  # malformed: not a receipt, never projected as one
    (False, ["capture_failed"], False),
])
def test_session_failure_custody_carries_its_attempts_own_checking_evidence(tmp_path, monkeypatch, fake_route,  # noqa: F811
                                                                            pending, evidence, carried):
    executor = _seat(tmp_path, monkeypatch, [(_poll_failure(evidence), pending)])
    with pytest.raises(ClaudexorUnavailable):
        executor.execute()
    facts = executor.failure_custody()
    assert (facts["pending_invocation_id"], facts["delegated_run_id"]) == ("inv-1" if pending else "", "run-2")
    if carried:
        assert facts["checking_evidence"] == evidence
    else:
        assert "checking_evidence" not in facts


def test_a_later_failure_without_evidence_never_reports_an_earlier_attempts_receipt(tmp_path, monkeypatch,
                                                                                    fake_route):  # noqa: F811
    executor = _seat(tmp_path, monkeypatch, [(_poll_failure(_GAP), True), (_poll_failure(), False)])
    for _ in range(2):
        with pytest.raises(ClaudexorUnavailable):
            executor.execute()
    facts = executor.failure_custody()
    assert facts["pending_invocation_id"] == "" and "checking_evidence" not in facts
    with pytest.raises(ClaudexorUnavailable):  # the settled failure replays without a new collection
        executor.execute()
    assert "checking_evidence" not in executor.failure_custody()


@pytest.mark.parametrize("own", [_RETAINED, None])
def test_a_collection_after_a_pending_gap_reports_its_own_evidence_when_its_verdict_fails(tmp_path, monkeypatch,
                                                                                          fake_route, own):  # noqa: F811
    """The retry after a pending failure that carried a typed gap settles a collection with its own receipt or none,
    then its verdict path raises: failure custody and the error-actor projection carry that collection's evidence,
    never the earlier attempt's gap."""
    from ouroboros.review_custody import _ReviewAttemptHistory, _review_exception_projection

    collected = {**_COLLECTED, **({"checking_evidence": own} if own else {})}
    executor = _seat(tmp_path, monkeypatch, [(_poll_failure(_GAP), True), (collected, False)])
    def bookkeeping_unavailable():
        raise RuntimeError("answer bookkeeping unavailable")
    executor.answer_observer = bookkeeping_unavailable
    with pytest.raises(ClaudexorUnavailable):
        executor.execute()
    assert executor.failure_custody()["checking_evidence"] == _GAP
    with pytest.raises(RuntimeError, match="answer bookkeeping unavailable") as failure:
        executor.execute()
    facts = executor.failure_custody()
    assert (facts["pending_invocation_id"], facts["delegated_run_id"]) == ("", "run-3")
    projected = _review_exception_projection(failure.value, facts, _ReviewAttemptHistory(), None)[0]
    assert facts.get("checking_evidence") == projected.get("checking_evidence") == own


def test_a_failed_collection_after_a_legacy_capture_gap_keeps_the_gap_and_never_reads_the_old_receipt(tmp_path):
    from ouroboros.review_session_custody import failed_start_seat_evidence

    gap = {"status": "capture_failed", "capture_gap": "legacy_successor_checking_copy_unavailable", "run_id": "run-2"}
    checking = {"legacy_capture_gap": gap, "allocation": {"allocated_by": "inv-1"}}
    evidence = failed_start_seat_evidence(tmp_path, "task-1", "op-1", RuntimeError("poll failed"), checking, "inv-1",
                                          unrequested=True, posted=False)
    assert evidence == gap and evidence is not gap and not list(tmp_path.iterdir())


def test_legacy_capture_generation_keeps_old_receipts_and_repeated_terminals_are_idempotent(repo, tmp_path, monkeypatch):
    from ouroboros import owner_keep
    from tests._review_session_route_shared import _terminal_detail

    task, operation = "a" * 32, "legacy-copy"
    drive = tmp_path / "drive"
    source = review_checking.freeze_checking_source(repo, data_root=drive, subject=review_checking.SUBJECT_INDEX)
    area = pathlib.Path(review_checking.prepare_seat_area(drive, task, operation, source))
    marker_path = area.parent / "checking.json"
    (area / "first.txt").write_text("predecessor")
    first = review_checking.capture_seat_evidence(drive, task, operation, hold=True)
    first_files = {p: p.read_bytes() for p in pathlib.Path(first["evidence_dir"]).rglob("*") if p.is_file()}
    for predecessor, run in (("run-1", "run-2"), ("run-2", "run-3")):
        token = "inv-" + run
        assert custody.record_start_requested(drive, task_id=task, invocation_id=token, idempotency_key=token,
            surface="multi_model_review", slot_id="seat", operation_id=operation,
            request={"continueFrom": predecessor}, continuation_of=predecessor)
        assert custody.record_started(drive, custody.RunCustody(run_id=run, task_id=task, invocation_id=token,
            continuation_of=predecessor), shape={"surface": "multi_model_review", "slot_id": "seat"})
    def advance(run):
        review_checking.prepare_legacy_successor_capture(
            drive, task, operation, run_id=run, execution_root=str(area), access="full")
    advance("run-2")
    marker = json.loads(marker_path.read_text())
    assert marker["legacy_run_id"] == "run-2" and "allocated_by" not in marker
    assert "held_for_continuation" not in marker
    (area / "second.txt").write_text("successor-two")
    second = review_checking.terminal_seat_evidence(drive, task, operation, _terminal_detail("failed", state="failed"),
                                                    hold=True)
    generation = marker_path.read_bytes()
    advance("run-2")  # repeated stopped terminal keeps its hold and its capture generation
    assert marker_path.read_bytes() == generation and area.exists()
    assert review_checking.capture_seat_evidence(drive, task, operation, hold=True) == second
    assert second["evidence_dir"] != first["evidence_dir"]
    assert all(path.read_bytes() == data for path, data in first_files.items())
    advance("run-3")
    assert json.loads(marker_path.read_text())["capture_id"] != marker["capture_id"]
    (area / "third.txt").write_text("successor-three")
    monkeypatch.setattr(owner_keep, "is_kept", lambda *_a: True)  # the gate still reads the owner's Keep
    third = review_checking.terminal_seat_evidence(drive, task, operation, _terminal_detail("ok"))
    assert third["status"] == "retained" and area.exists()
    assert b"+successor-three\n" in (pathlib.Path(third["evidence_dir"]) / "delta.patch").read_bytes()
    saved_marker = marker_path.read_bytes()
    advance("run-3")
    advance("run-2")  # stale callback of the same operation cannot roll the capture backwards
    assert marker_path.read_bytes() == saved_marker
    assert review_checking.terminal_seat_evidence(drive, task, operation, _terminal_detail("ok")) == third
    monkeypatch.setattr(owner_keep, "is_kept", lambda *_a: False)
    review_checking.terminal_seat_evidence(drive, task, operation, _terminal_detail("ok"))
    assert not area.exists()
    advance("run-3")  # duplicate callback after release neither remakes nor advances it
    assert marker_path.read_bytes() == saved_marker


@pytest.mark.parametrize("gap", ["holder", "root", "marker", "missing_root"])
def test_legacy_capture_generation_never_takes_another_holder_or_invents_a_writable_copy(repo, tmp_path, gap):
    task, operation = "b" * 32, "legacy-copy"
    source = review_checking.freeze_checking_source(repo, data_root=tmp_path, subject=review_checking.SUBJECT_INDEX)
    area = pathlib.Path(review_checking.prepare_seat_area(tmp_path, task, operation, source))
    marker = area.parent / "checking.json"
    if gap == "holder":
        facts = json.loads(marker.read_text())
        marker.write_text(json.dumps({**facts, "holder_operation_id": "newer-operation"}))
    if gap == "marker":
        marker.rename(marker.with_suffix(".saved"))  # recoverable missing marker, not a read-only seat
    before = marker.read_bytes() if marker.exists() else None
    with pytest.raises(review_checking.CheckingAreaUnavailable):
        review_checking.prepare_legacy_successor_capture(tmp_path, task, operation, run_id="old-callback",
            execution_root="" if gap == "missing_root" else str(area / "wrong") if gap == "root" else str(area),
            access="full")
    assert (marker.read_bytes() if marker.exists() else None) == before and area.exists()
