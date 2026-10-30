"""Host consumer tests for delegated reviewer threads and retained checking output.

The real session consumer, custody, checking-copy materialization and evidence
capture run here. Only the engine transport is synthetic; no native harness,
daemon, credentials or paid provider is used.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from ouroboros import claudexor_daemon
from ouroboros import review_checking as rc
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.review_execution import (
    ReviewRouteUnavailable,
    SessionInvocation,
    review_session_route,
    run_delegated_review_session,
)
from tests._review_session_route_shared import (  # noqa: F401
    FakeGateway,
    FakeLLM,
    _agent_request,
    _agent_slot,
    fake_route,
)
from tests.test_review_checking import _author_state, _git, author  # noqa: F401
from tests.test_review_checkout_lifecycle import ABSENT, NO_CENSUS, SETTLED, UNSETTLED, directory_source


class ThreadGateway(FakeGateway):
    """The public thread API backed by physical files in the host checking area."""

    def __init__(self):
        super().__init__()
        self.engine_version = "3.23.0"
        self.thread_requests = []
        self.turn_requests = []
        self.turns = []
        self.trust = []
        self.terminal_state = "succeeded"
        self.failure_code = "reviewer_probe_failed"
        self.expected_subject = b"A staged\n"
        self.before_turn = None
        self.handles = {}
        self.lose_start_response = False
        self.process_scope = ABSENT  # summary.processScope; an older engine omits the key

    def ensure_full_access(self, root):
        self.trust.append(root)
        return {"allowFullAccess": True}

    def create_thread(self, request, *, idempotency_key):
        self.thread_requests.append(dict(request))
        self.thread = {"id": "thread-1", "headRunId": ""}
        return {"id": "thread-1"}

    def start_thread_turn(self, thread_id, request, *, idempotency_key):
        assert thread_id == "thread-1"
        self.turn_requests.append((thread_id, dict(request), idempotency_key))
        if idempotency_key in self.handles:
            return dict(self.handles[idempotency_key])
        number = len(self.handles) + 1
        binding = self.thread_requests[0]
        if binding["access"] != "readonly":
            assert binding["workspace"] == "delegated"
            area = Path(binding["workspaceRoot"])
            assert area.is_dir() and area != Path(binding["scope"]["root"])
            assert (area / "f.txt").read_bytes() == self.expected_subject
            if self.before_turn:
                self.before_turn(area)
            (area / "reviewer-test.txt").write_bytes(f"reviewer turn {number}\n".encode())
            declared = area / rc.EVIDENCE_DIRNAME
            declared.mkdir(exist_ok=True)
            (declared / "coverage.xml").write_bytes(self.coverage(number))
            (declared / f"turn-{number}.log").write_bytes(f"terminal turn {number}\n".encode())
        run_id, turn_id = f"run-{number}", f"turn-{number}"
        self.turns.append({"id": turn_id, "runId": run_id,
                           "continuity": {"kind": "native_resume" if number > 1 else "new_session"}})
        self.thread["headRunId"] = run_id
        handle = self.handles[idempotency_key] = {"runId": run_id, "turnId": turn_id}
        if self.lose_start_response:
            self.lose_start_response = False
            raise TimeoutError("turn was accepted; response was lost")
        return dict(handle)

    @staticmethod
    def coverage(number):
        return f'<coverage turn="{number}"/>\r\n'.encode() + b"\x00\xff"

    def get_run(self, run_id, **kwargs):
        detail = super().get_run(run_id, **kwargs)
        detail["summary"]["state"] = self.terminal_state
        if self.process_scope is not ABSENT:
            detail["summary"]["processScope"] = self.process_scope
        if self.terminal_state != "succeeded":
            detail["summary"]["failure"] = {
                "code": self.failure_code, "safeMessage": "The reviewer probe failed",
            }
        return detail

    def get_thread(self, thread_id):
        assert thread_id == "thread-1"
        return {"thread": dict(self.thread), "turns": list(self.turns),
                "sessions": [{"id": "native-session-1", "harnessId": "fake-review",
                              "profileId": "", "state": "ready"}]}


@pytest.fixture
def thread_gateway(fake_route, monkeypatch):  # noqa: F811
    FakeGateway.catalog_entry["accessProfilesSupported"] = ["readonly", "workspace_write", "full"]
    gateway = ThreadGateway()
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: gateway)
    return gateway


def _run(author, drive, source, *, operation="op-1", access="full", thread_id="", retry_state=None, **kwargs):  # noqa: F811
    fields = dict(
        task_id="task-1", surface="plan_review", slot_id="t1", timeout_sec=60,
        session_route=review_session_route(), retry_state=retry_state if retry_state is not None else {},
        operation_id=operation, use_thread=True, thread_id=thread_id, access=access, checking_source=source,
    )
    fields.update(kwargs)
    return run_delegated_review_session(
        prompt=f"Review the frozen subject in {operation}", root=str(author), custody_drive=drive,
        invocation=SessionInvocation(**fields),
    )


def _assert_coverage(evidence, expected):
    assert evidence["status"] == "retained"
    name = f"{rc.EVIDENCE_DIRNAME}/coverage.xml"
    entry = next(row for row in evidence["named_evidence"] if row["path"] == name)
    assert entry["size"] == len(expected)
    assert entry["sha256"] == hashlib.sha256(expected).hexdigest()
    assert (Path(evidence["evidence_dir"]) / "files" / name).read_bytes() == expected


@pytest.mark.parametrize("access", ["workspace_write", "full"])
def test_successful_turns_refresh_subject_in_same_thread_and_retain_each_output(author, tmp_path, thread_gateway, access):  # noqa: F811
    """Real host functions consume settled success, capture, and refresh automatically.

    The synthetic transport exercises the canonical exact-run success boundary;
    live native adapter qualification remains a separate parent-owned check.
    """
    drive = tmp_path / "drive"
    source_a = rc.freeze_checking_source(author, data_root=drive, subject=rc.SUBJECT_INDEX)
    before_a = _author_state(author)
    first = _run(author, drive, source_a, access=access)
    area = rc.seat_dir(drive, "task-1", "thread:plan_review:t1") / "area.noindex"
    assert area.is_dir()
    _assert_coverage(first["checking_evidence"], ThreadGateway.coverage(1))
    first_summary = (Path(first["checking_evidence"]["evidence_dir"]) / "summary.json").read_bytes()
    assert _author_state(author) == before_a

    # The author prepares a distinct staged subject and ignored input between
    # rounds. The next real host invocation refreshes the same bound root.
    (author / "f.txt").write_bytes(b"B staged\n")
    _git(author, "add", "f.txt")
    (author / "f.txt").write_bytes(b"B staged\nC unstaged\n")
    (author / ".review-drive" / "task" / "view.txt").write_bytes(b"second-round input\x00\xff")
    source_b = rc.freeze_checking_source(author, data_root=drive, subject=rc.SUBJECT_INDEX)
    assert source_b["tree"] != source_a["tree"]
    before_b = _author_state(author)
    thread_gateway.expected_subject = b"B staged\n"
    seen_roots = []

    def inspect_refreshed_subject(current):
        seen_roots.append(current)
        assert current == area
        assert _git(current, "write-tree") == source_b["tree"]
        assert _git(current, "rev-parse", "HEAD") == source_b["parent"]
        assert (current / ".review-drive" / "task" / "view.txt").read_bytes() == b"second-round input\x00\xff"
        assert not (current / "reviewer-test.txt").exists()
        assert not (current / rc.EVIDENCE_DIRNAME / "turn-1.log").exists()

    thread_gateway.before_turn = inspect_refreshed_subject
    second = _run(author, drive, source_b, operation="op-2", access=access,
                  thread_id=first["thread_id"])
    assert seen_roots == [area]
    _assert_coverage(second["checking_evidence"], ThreadGateway.coverage(2))
    _assert_coverage(first["checking_evidence"], ThreadGateway.coverage(1))
    assert (Path(first["checking_evidence"]["evidence_dir"]) / "summary.json").read_bytes() == first_summary
    assert first["checking_evidence"]["evidence_dir"] != second["checking_evidence"]["evidence_dir"]
    for number, facts in enumerate((first, second), 1):
        saved_log = Path(facts["checking_evidence"]["evidence_dir"]) / "files" / rc.EVIDENCE_DIRNAME / f"turn-{number}.log"
        assert saved_log.read_bytes() == f"terminal turn {number}\n".encode()
    assert first["thread_id"] == second["thread_id"] == "thread-1"
    assert first["turn_id"] == "turn-1" and second["turn_id"] == "turn-2"
    assert first["thread_receipt"]["sessions"] == second["thread_receipt"]["sessions"]
    assert first["thread_receipt"]["sessions"][0]["id"] == "native-session-1"
    assert second["thread_receipt"]["continuity"] == {"kind": "native_resume"}
    assert area.is_dir() and _author_state(author) == before_b
    assert len(thread_gateway.thread_requests) == 1 and not thread_gateway.start_requests
    create = thread_gateway.thread_requests[0]
    assert create["scope"] == {"kind": "project", "root": str(author)}
    assert create["workspace"] == "delegated" and create["workspaceRoot"] == str(area)
    assert create["mode"] == "agent" and create["access"] == access
    assert thread_gateway.trust == ([str(author), str(author)] if access == "full" else [])
    assert all(request["execution"] == {"isolation": "live"} and "scope" not in request
               for _, request, _ in thread_gateway.turn_requests)


@pytest.mark.parametrize("access", ["workspace_write", "full"])
def test_directory_turns_keep_geometry_same_thread_and_frozen_evidence(tmp_path, thread_gateway, access):
    from ouroboros import delegate_custody as custody

    source_dir, drive = tmp_path / "ordinary folder", tmp_path / "drive"
    source_dir.mkdir()
    (source_dir / "f.txt").write_bytes(b"A staged\n")
    source = directory_source(source_dir, drive)
    first = _run(source_dir, drive, source, access=access)
    area = rc.seat_dir(drive, "task-1", "thread:plan_review:t1") / "area.noindex"
    receipt = Path(first["checking_evidence"]["evidence_dir"]) / "summary.json"
    preserved = receipt.read_bytes()
    assert source["kind"] == "directory" and not (area / ".git").exists()
    _assert_coverage(first["checking_evidence"], ThreadGateway.coverage(1))
    assert (source_dir / "f.txt").read_bytes() == b"A staged\n"

    (source_dir / "f.txt").write_bytes(b"B revised\n")
    thread_gateway.expected_subject = b"B revised\n"
    second_source = directory_source(source_dir, drive)
    thread_gateway.before_turn = lambda current: (
        current == area and not (current / "reviewer-test.txt").exists()
        and not (current / ".git").exists()) or pytest.fail("directory turn did not refresh its bound copy")
    second = _run(source_dir, drive, second_source, operation="op-2", access=access,
                  thread_id=first["thread_id"])
    assert first["thread_id"] == second["thread_id"] == "thread-1"
    assert second["turn_id"] == "turn-2" and len(thread_gateway.thread_requests) == 1
    assert len(thread_gateway.turn_requests) == 2 and not thread_gateway.start_requests
    for _, wire, key in thread_gateway.turn_requests:
        assert wire["execution"] == {"workspaceKind": "directory", "isolation": "live"}
        stored = custody.invocation_record(drive, key)["request"]
        assert stored["execution"] == {"workspaceKind": "directory", "isolation": "live",
                                       "delegated": True, "workspaceRoot": str(area)}
        assert stored["access"] == wire["access"] == access
    assert receipt.read_bytes() == preserved
    _assert_coverage(second["checking_evidence"], ThreadGateway.coverage(2))
    assert sorted(path.name for path in source_dir.iterdir()) == ["f.txt"]
    assert (source_dir / "f.txt").read_bytes() == b"B revised\n"


@pytest.mark.parametrize("scope", [SETTLED, UNSETTLED], ids=["settled", "unsettled"])
def test_failed_directory_turn_refreshes_after_its_capture_whatever_the_scope(tmp_path, thread_gateway, scope):
    source_dir, drive = tmp_path / "folder", tmp_path / "drive"
    source_dir.mkdir()
    (source_dir / "f.txt").write_bytes(b"A staged\n")
    source = directory_source(source_dir, drive)

    def failed_preparation(area):
        (area / ".git").mkdir()
        (area / ".git" / "index.lock").write_bytes(b"")

    thread_gateway.before_turn = failed_preparation
    thread_gateway.terminal_state = "failed"
    thread_gateway.failure_code = "workspace_unavailable"
    thread_gateway.process_scope = scope
    with pytest.raises(ClaudexorUnavailable) as caught:
        _run(source_dir, drive, source)
    assert caught.value.code == "workspace_unavailable"
    evidence = caught.value.checking_evidence
    assert evidence["status"] == "retained" and "refresh_safe" not in evidence
    captured = Path(evidence["evidence_dir"]) / "summary.json"
    preserved = captured.read_bytes()
    area = rc.seat_dir(drive, "task-1", "thread:plan_review:t1") / "area.noindex"
    assert (area / ".git" / "index.lock").is_file()  # a thread keeps its copy between turns
    thread_gateway.terminal_state = "succeeded"
    thread_gateway.before_turn = lambda current: (
        current == area and not (current / ".git").exists()
    ) or pytest.fail("refresh carried the failed preparation's synthetic Git directory")
    second = _run(source_dir, drive, source, operation="op-2", thread_id="thread-1")
    assert second["thread_id"] == "thread-1" and second["turn_id"] == "turn-2"
    assert thread_gateway.turn_requests[-1][1]["execution"]["workspaceKind"] == "directory"
    assert len(thread_gateway.turn_requests) == 2
    assert captured.read_bytes() == preserved
    assert sorted(path.name for path in source_dir.iterdir()) == ["f.txt"]


@pytest.mark.parametrize("state,failure_code", [
    ("failed", "reviewer_probe_failed"),
    ("cancelled", "reviewer_probe_failed"),
    ("interrupted", "reviewer_probe_failed"),
    ("interrupted", "host_restart"),
])
def test_failed_terminal_turn_keeps_named_output_and_original_error(author, tmp_path, thread_gateway, state, failure_code):  # noqa: F811
    drive = tmp_path / "drive"
    source = rc.freeze_checking_source(author, data_root=drive, subject=rc.SUBJECT_INDEX)
    before = _author_state(author)
    thread_gateway.terminal_state = state
    thread_gateway.failure_code = failure_code
    with pytest.raises(ClaudexorUnavailable) as caught:
        _run(author, drive, source)
    assert caught.value.code == failure_code
    assert caught.value.delegated_run_id == "run-1"
    evidence = caught.value.checking_evidence
    _assert_coverage(evidence, ThreadGateway.coverage(1))
    assert not {"refresh_safe", "retention_reason"} & set(evidence)
    thread_gateway.terminal_state = "succeeded"
    second = _run(author, drive, source, operation="op-2", thread_id="thread-1")
    assert second["turn_id"] == "turn-2" and len(thread_gateway.turn_requests) == 2
    _assert_coverage(evidence, ThreadGateway.coverage(1))
    _assert_coverage(second["checking_evidence"], ThreadGateway.coverage(2))
    assert _author_state(author) == before


@pytest.mark.parametrize("scope", [
    ABSENT, SETTLED, NO_CENSUS, None, UNSETTLED, {**NO_CENSUS, "survivors": 1}, {**NO_CENSUS, "unreadable": 1},
    {**NO_CENSUS, "coverage": "marker"},
], ids=["absent", "settled", "no-census", "null", "unsettled", "no-census-survivor", "no-census-unreadable",
        "mixed-coverage"])
def test_any_process_scope_lets_the_next_turn_refresh_the_copy(author, tmp_path, thread_gateway, scope):  # noqa: F811
    """The answer and evidence of a succeeded turn are delivered whatever process census
    the engine reports, and the next turn starts on a fresh copy (owner 1A, 2026-10-12)."""
    drive = tmp_path / "drive"
    source = rc.freeze_checking_source(author, data_root=drive, subject=rc.SUBJECT_INDEX)
    before = _author_state(author)
    thread_gateway.process_scope = scope
    left_over = []
    thread_gateway.before_turn = lambda area: left_over.append((area / "reviewer-test.txt").exists())
    first = _run(author, drive, source)
    assert first["text"] == '{"findings": []}' and first["run_id"] == "run-1"
    _assert_coverage(first["checking_evidence"], ThreadGateway.coverage(1))
    assert not {"refresh_safe", "release_basis", "process_scope"} & set(first["checking_evidence"])
    second = _run(author, drive, source, operation="op-2", thread_id=first["thread_id"])
    assert second["turn_id"] == "turn-2" and left_over == [False, False], "turn 2 starts on a fresh copy"
    _assert_coverage(second["checking_evidence"], ThreadGateway.coverage(2))
    assert len(thread_gateway.turn_requests) == 2
    assert len(thread_gateway.thread_requests) == 1 and not thread_gateway.start_requests
    assert not thread_gateway.cancels and _author_state(author) == before


def test_engine_upgrade_between_turns_reuses_the_stored_receipt(author, tmp_path, thread_gateway):  # noqa: F811
    drive = tmp_path / "drive"
    source = rc.freeze_checking_source(author, data_root=drive, subject=rc.SUBJECT_INDEX)
    first = _run(author, drive, source)  # an older engine: no projection
    stored = Path(first["checking_evidence"]["evidence_dir"]) / "summary.json"
    stored_bytes = stored.read_bytes()
    thread_gateway.process_scope = None  # upgraded engine: no readable scope record of run-1 or run-2
    assert rc.terminal_seat_evidence(drive, "task-1", "op-1", thread_gateway.get_run("run-1")) == first["checking_evidence"]
    assert stored.read_bytes() == stored_bytes
    second = _run(author, drive, source, operation="op-2", thread_id=first["thread_id"])
    third = _run(author, drive, source, operation="op-3", thread_id=first["thread_id"])
    assert (second["turn_id"], third["turn_id"]) == ("turn-2", "turn-3")
    assert len(thread_gateway.turn_requests) == 3 and stored.read_bytes() == stored_bytes


def test_unobserved_terminal_is_captured_by_the_next_turn(author, tmp_path, thread_gateway, monkeypatch):  # noqa: F811
    from ouroboros import review_execution as rx

    drive = tmp_path / "drive"
    source = rc.freeze_checking_source(author, data_root=drive, subject=rc.SUBJECT_INDEX)
    original_error = OSError("daemon connection lost before terminal receipt")
    real_poll, failed = rx._poll_session_terminal, []

    def lost_once(*args, **kwargs):
        if not failed:
            failed.append(True)
            raise original_error
        return real_poll(*args, **kwargs)

    monkeypatch.setattr(rx, "_poll_session_terminal", lost_once)
    with pytest.raises(OSError) as caught:
        _run(author, drive, source)
    assert caught.value is original_error
    assert not getattr(caught.value, "checking_evidence", None)
    area = rc.seat_dir(drive, "task-1", "thread:plan_review:t1") / "area.noindex"
    assert (area / rc.EVIDENCE_DIRNAME / "coverage.xml").read_bytes() == ThreadGateway.coverage(1)
    second = _run(author, drive, source, operation="op-2", thread_id="thread-1")
    assert second["turn_id"] == "turn-2" and len(thread_gateway.turn_requests) == 2
    _assert_coverage(rc._receipt(drive, "task-1", "op-1"), ThreadGateway.coverage(1))
    _assert_coverage(second["checking_evidence"], ThreadGateway.coverage(2))


def test_capture_failure_keeps_the_answer_and_does_not_block_the_thread(author, tmp_path, thread_gateway, monkeypatch):  # noqa: F811
    drive = tmp_path / "drive"
    source = rc.freeze_checking_source(author, data_root=drive, subject=rc.SUBJECT_INDEX)

    def unavailable(*args, **kwargs):
        raise OSError("evidence volume unavailable")

    monkeypatch.setattr(rc, "_git_delta", unavailable)
    first = _run(author, drive, source)
    assert first["text"] == '{"findings": []}'
    assert first["checking_evidence"]["status"] == "capture_failed"
    assert "evidence volume unavailable" in first["checking_evidence"]["error"]
    second = _run(author, drive, source, operation="op-2", thread_id=first["thread_id"])
    assert second["turn_id"] == "turn-2" and not thread_gateway.start_requests
    assert rc._receipt(drive, "task-1", "op-1")["status"] == "capture_failed"


@pytest.mark.parametrize("ordinary_directory", [False, True], ids=["git", "directory"])
def test_lost_turn_response_rejoins_original_copy_and_invocation(author, tmp_path, thread_gateway, ordinary_directory):  # noqa: F811
    from ouroboros import delegate_custody as custody

    drive = tmp_path / "drive"
    root = tmp_path / "ordinary" if ordinary_directory else author
    if ordinary_directory:
        root.mkdir()
        (root / "f.txt").write_bytes(b"A staged\n")
    source = (directory_source(root, drive) if ordinary_directory
              else rc.freeze_checking_source(root, data_root=drive, subject=rc.SUBJECT_INDEX))
    before, retry_state = _author_state(author), {}
    thread_gateway.lose_start_response = True
    with pytest.raises(TimeoutError):
        _run(root, drive, source, retry_state=retry_state)
    area = rc.seat_dir(drive, "task-1", "thread:plan_review:t1") / "area.noindex"
    assert (area / rc.EVIDENCE_DIRNAME / "coverage.xml").read_bytes() == ThreadGateway.coverage(1)
    pending = retry_state["pending_invocation_id"]
    recorded = custody.invocation_record(drive, pending)["request"]
    facts = _run(root, drive, None, retry_state=retry_state, reconcile_only=True)
    _assert_coverage(facts["checking_evidence"], ThreadGateway.coverage(1))
    assert facts["idempotent_recovery"] and facts["run_id"] == "run-1"
    assert len(thread_gateway.thread_requests) == len(thread_gateway.turns) == 1
    assert len(thread_gateway.turn_requests) == 2
    assert [row[2] for row in thread_gateway.turn_requests] == [pending, pending]
    assert thread_gateway.turn_requests[0] == thread_gateway.turn_requests[1]
    assert custody.invocation_record(drive, pending)["request"] == recorded
    assert recorded["execution"].get("workspaceKind") == ("directory" if ordinary_directory else None)
    assert not thread_gateway.start_requests
    assert _author_state(author) == before and area.is_dir()
    if ordinary_directory:
        assert sorted(path.name for path in root.iterdir()) == ["f.txt"]
        assert (root / "f.txt").read_bytes() == b"A staged\n"


def test_pre_floor_writable_thread_refuses_without_a_fresh_run(author, tmp_path, thread_gateway):  # noqa: F811
    drive = tmp_path / "drive"
    source = rc.freeze_checking_source(author, data_root=drive, subject=rc.SUBJECT_INDEX)
    thread_gateway.engine_version = "3.22.0"
    paid = []
    with pytest.raises(ReviewRouteUnavailable) as caught:
        _run(author, drive, source, dispatch_stamp=lambda: paid.append(True))
    assert caught.value.code == "review_checking_thread_unsupported"
    assert not paid
    assert not thread_gateway.registrations and not thread_gateway.project_lookups
    assert not thread_gateway.trust
    assert not rc.seat_dir(drive, "task-1", "thread:plan_review:t1").exists()
    assert not thread_gateway.thread_requests and not thread_gateway.turn_requests
    assert not thread_gateway.start_requests


@pytest.mark.parametrize("version", ["3.22.0", "3.23.0"])
def test_plan_surface_executor_keeps_writable_thread_semantics(author, tmp_path, thread_gateway, version):  # noqa: F811
    """The caller must not silently replace sticky continuity with a fresh run."""
    from ouroboros.review_substrate import run_review_request

    drive = tmp_path / "drive"
    source = rc.freeze_checking_source(author, data_root=drive, subject=rc.SUBJECT_INDEX)
    thread_gateway.engine_version = version
    result = run_review_request(
        _agent_request(surface="plan_review", call_type="plan_review", task_id="task-1",
                       session_root=str(author), checking_source=source),
        slots=[_agent_slot(slot_id="t1", session_access="full")],
        drive_root=drive, llm=FakeLLM(),
    )
    assert not thread_gateway.start_requests
    if version == "3.23.0":
        assert len(thread_gateway.thread_requests) == len(thread_gateway.turn_requests) == 1
        assert result.actors[0]["usage"]["review_thread_id"] == "thread-1"
    else:
        assert not thread_gateway.thread_requests and not thread_gateway.turn_requests
        assert result.actors[0]["status"] != "ok"


def test_readonly_thread_keeps_legacy_binding(author, tmp_path, thread_gateway):  # noqa: F811
    thread_gateway.engine_version = "3.3.7"
    before = _author_state(author)
    first = _run(author, tmp_path / "drive", None, access="readonly")
    second = _run(author, tmp_path / "drive", None, access="readonly", operation="op-2",
                  thread_id=first["thread_id"])
    assert first["thread_id"] == second["thread_id"] == "thread-1"
    assert len(thread_gateway.thread_requests) == 1 and len(thread_gateway.turn_requests) == 2
    create = thread_gateway.thread_requests[0]
    assert create["scope"] == {"kind": "project", "root": str(author)}
    assert create["mode"] == "ask" and create["access"] == "readonly"
    assert create["workspace"] == "in_place" and "workspaceRoot" not in create
    assert all("execution" not in request for _, request, _ in thread_gateway.turn_requests)
    assert not thread_gateway.trust and not thread_gateway.start_requests
    assert not first["checking_evidence"] and not second["checking_evidence"]
    assert _author_state(author) == before


def test_directory_capture_retains_named_coverage_bytes_and_digest(tmp_path):
    source_folder, drive = tmp_path / "payload", tmp_path / "drive"
    source_folder.mkdir()
    (source_folder / "SKILL.md").write_bytes(b"original skill\n")
    source = directory_source(source_folder, drive)
    assert source["kind"] == "directory"
    area = Path(rc.prepare_seat_area(drive, "task-1", "directory-op", source))
    declared = area / rc.EVIDENCE_DIRNAME
    declared.mkdir()
    expected = b'<coverage branch-rate="0.7"/>\r\n\x00\xff'
    (declared / "coverage.xml").write_bytes(expected)
    # This test itself completed the only synchronous writer before capture.
    evidence = rc.capture_seat_evidence(drive, "task-1", "directory-op")
    _assert_coverage(evidence, expected)
    assert not area.exists()
    assert (source_folder / "SKILL.md").read_bytes() == b"original skill\n"
    assert not (source_folder / rc.EVIDENCE_DIRNAME).exists()
    saved = json.loads((Path(evidence["evidence_dir"]) / "summary.json").read_text())
    assert saved["named_evidence"] == evidence["named_evidence"]


def test_a_captured_thread_turn_keeps_its_copy_until_the_next_turn(author, tmp_path):  # noqa: F811
    drive = tmp_path / "drive"
    source = rc.freeze_checking_source(author, data_root=drive, subject=rc.SUBJECT_INDEX)
    area = Path(rc.thread_seat_area(drive, "task-1", "plan_review:t1", source, operation_id="op-1"))
    declared = area / rc.EVIDENCE_DIRNAME
    declared.mkdir()
    (declared / "coverage.xml").write_bytes(ThreadGateway.coverage(1))
    evidence = rc.capture_seat_evidence(drive, "task-1", "op-1")
    _assert_coverage(evidence, ThreadGateway.coverage(1))
    assert (declared / "coverage.xml").read_bytes() == ThreadGateway.coverage(1)
    assert rc.thread_seat_area(drive, "task-1", "plan_review:t1", source, operation_id="op-2") == str(area)
    assert not declared.exists()
    saved = Path(evidence["evidence_dir"]) / "files" / rc.EVIDENCE_DIRNAME / "coverage.xml"
    assert saved.read_bytes() == ThreadGateway.coverage(1)


def test_saved_host_thread_id_resumes_when_caller_supplies_none(author, tmp_path, thread_gateway):  # noqa: F811
    """Normal success opens the next turn without an external release call."""
    drive = tmp_path / "drive"
    source = rc.freeze_checking_source(author, data_root=drive, subject=rc.SUBJECT_INDEX)
    before = _author_state(author)
    first = _run(author, drive, source)
    home = rc.seat_dir(drive, "task-1", "thread:plan_review:t1")
    marker = json.loads((home / "checking.json").read_text())
    assert marker["thread_id"] == first["thread_id"] == "thread-1"
    second = _run(author, drive, source, operation="op-2")
    assert second["thread_id"] == first["thread_id"]
    assert second["turn_id"] == "turn-2"
    assert len(thread_gateway.thread_requests) == 1 and len(thread_gateway.turn_requests) == 2
    assert not thread_gateway.start_requests
    assert all(thread_id == "thread-1" for thread_id, _, _ in thread_gateway.turn_requests)
    assert thread_gateway.thread_requests[0]["workspaceRoot"] == str(home / "area.noindex")
    _assert_coverage(first["checking_evidence"], ThreadGateway.coverage(1))
    _assert_coverage(second["checking_evidence"], ThreadGateway.coverage(2))
    assert _author_state(author) == before


def test_mismatched_supplied_thread_id_refuses_before_refresh_or_send(author, tmp_path, thread_gateway):  # noqa: F811
    drive = tmp_path / "drive"
    source = rc.freeze_checking_source(author, data_root=drive, subject=rc.SUBJECT_INDEX)
    before = _author_state(author)
    first = _run(author, drive, source)
    home = rc.seat_dir(drive, "task-1", "thread:plan_review:t1")
    area = home / "area.noindex"
    # The turn's receipt exists; an identity mismatch still prevents the refresh.
    assert first["checking_evidence"]["status"] == "retained"
    marker_before = (home / "checking.json").read_bytes()
    marker_inode = area.stat().st_ino
    paid = []
    with pytest.raises(ReviewRouteUnavailable) as caught:
        _run(author, drive, source, operation="op-2", thread_id="different-thread",
             dispatch_stamp=lambda: paid.append(True))
    assert caught.value.code == "review_checking_copy_unavailable"
    assert not paid
    assert len(thread_gateway.thread_requests) == len(thread_gateway.turn_requests) == 1
    assert not thread_gateway.start_requests
    assert (home / "checking.json").read_bytes() == marker_before
    assert area.stat().st_ino == marker_inode
    assert (area / "reviewer-test.txt").read_bytes() == b"reviewer turn 1\n"
    assert (area / rc.EVIDENCE_DIRNAME / "coverage.xml").read_bytes() == ThreadGateway.coverage(1)
    _assert_coverage(first["checking_evidence"], ThreadGateway.coverage(1))
    assert _author_state(author) == before


def test_cancelled_terminal_carries_evidence_on_original_poller_timeout(author, tmp_path, thread_gateway, monkeypatch):  # noqa: F811
    from ouroboros import review_execution as rx

    drive = tmp_path / "drive"
    source = rc.freeze_checking_source(author, data_root=drive, subject=rc.SUBJECT_INDEX)
    before = _author_state(author)
    original_get, original_cancel = thread_gateway.get_run, thread_gateway.cancel_run

    def get_run(run_id, **kwargs):
        if not thread_gateway.cancels:
            thread_gateway.run_gets.append(run_id)
            return {"summary": {"state": "running"}, "lastSeq": 1}
        return original_get(run_id, **kwargs)

    def cancel_run(run_id, *, reason=""):
        receipt = original_cancel(run_id, reason=reason)
        thread_gateway.terminal_state = "cancelled"
        return receipt

    poll_errors = []
    original_poll = rx._poll_session_terminal

    def observe_poller(*args, **kwargs):
        try:
            return original_poll(*args, **kwargs)
        except TimeoutError as error:
            poll_errors.append(error)
            raise

    monkeypatch.setattr(thread_gateway, "get_run", get_run)
    monkeypatch.setattr(thread_gateway, "cancel_run", cancel_run)
    monkeypatch.setattr(rx, "_poll_session_terminal", observe_poller)
    with pytest.raises(TimeoutError, match="exceeded the slot budget") as caught:
        _run(author, drive, source, timeout_sec=0)
    assert poll_errors == [caught.value]
    assert caught.value is poll_errors[0]
    assert caught.value.delegated_run_id == "run-1"
    assert caught.value.checking_terminal_detail["summary"]["state"] == "cancelled"
    assert thread_gateway.cancels == [("run-1", "review_slot_timeout")]
    _assert_coverage(caught.value.checking_evidence, ThreadGateway.coverage(1))
    area = rc.seat_dir(drive, "task-1", "thread:plan_review:t1") / "area.noindex"
    assert (area / rc.EVIDENCE_DIRNAME / "coverage.xml").read_bytes() == ThreadGateway.coverage(1)
    assert _author_state(author) == before


@pytest.mark.skipif(os.name == "nt", reason="symlink privileges vary on Windows")
def test_completed_turn_evidence_remains_readable_when_later_area_is_replaced(author, tmp_path):  # noqa: F811
    """Collection reads the original immutable artifact without requiring its live area."""
    drive = tmp_path / "drive"
    source = rc.freeze_checking_source(author, data_root=drive, subject=rc.SUBJECT_INDEX)
    area = Path(rc.thread_seat_area(drive, "task-1", "plan_review:t1", source, operation_id="op-1"))
    declared = area / rc.EVIDENCE_DIRNAME
    declared.mkdir()
    (declared / "coverage.xml").write_bytes(b"original-turn\x00")
    original = rc.capture_seat_evidence(drive, "task-1", "op-1")
    area.rename(area.with_name("retained-away"))
    area.symlink_to(author, target_is_directory=True)
    assert rc.capture_seat_evidence(drive, "task-1", "op-1") == original
    _assert_coverage(original, b"original-turn\x00")


def test_started_writable_thread_collects_when_engine_cannot_start_new_threads(
    author, tmp_path, thread_gateway, monkeypatch,  # noqa: F811
):
    from ouroboros import review_execution

    drive = tmp_path / "drive"
    source = rc.freeze_checking_source(author, data_root=drive, subject=rc.SUBJECT_INDEX)
    state = {}
    original_poll = review_execution._poll_session_terminal
    monkeypatch.setattr(review_execution, "_poll_session_terminal", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("transport interrupted after durable run start")))
    with pytest.raises(RuntimeError, match="transport interrupted"):
        _run(author, drive, source, retry_state=state)
    assert state["delegated_run_id"] == "run-1"
    thread_gateway.engine_version = "3.22.0"
    monkeypatch.setattr(review_execution, "_poll_session_terminal", original_poll)
    result = _run(author, drive, source, retry_state=state)
    assert result["idempotent_recovery"] and result["run_id"] == "run-1"
    _assert_coverage(result["checking_evidence"], ThreadGateway.coverage(1))
    assert len(thread_gateway.thread_requests) == len(thread_gateway.turn_requests) == 1
    assert not thread_gateway.start_requests


@pytest.mark.parametrize("access", ["readonly", "full"])
def test_writable_plan_seat_packet_keeps_the_snapshot_source_rule(author, tmp_path, thread_gateway, access):  # noqa: F811
    """Technical access widens what a seat may run in its copy, never what it may read.

    The actual thread turn of a plan seat carries the plan runtime's shared-snapshot
    task text; a writable seat's checking-copy instructions restate that rule instead of
    a bare 'retrieve the evidence yourself', while a read-only seat keeps its legacy
    read-only role. Whether a live harness can resolve each named source from its copy
    is a native qualification, not shown by this synthetic transport."""
    from ouroboros.review_substrate import run_review_request
    from ouroboros.tools.plan_review_runtime import _session_task_text

    drive = tmp_path / "drive"
    source = rc.freeze_checking_source(author, data_root=drive, subject=rc.SUBJECT_INDEX) if access != "readonly" else None
    task = _session_task_text("PLAN REVIEW SYSTEM", "EVIDENCE (redacted): owner said [REDACTED]", str(author))
    result = run_review_request(
        _agent_request(surface="plan_review", call_type="plan_review", task_id="task-1",
                       session_root=str(author), checking_source=source, session_task=task),
        slots=[_agent_slot(slot_id="t1", session_access=access)], drive_root=drive, llm=FakeLLM(),
    )
    assert result.actors[0]["usage"]["review_thread_id"] == "thread-1"
    [(_thread, turn, _key)] = thread_gateway.turn_requests
    assert "the same bytes every reviewer sees" in turn["prompt"]
    assert "owner said [REDACTED]" in turn["prompt"]
    if access == "readonly":
        assert turn["instructions"].startswith("You are a delegated read-only REVIEWER session.")
        assert "CHECKING COPY" not in turn["instructions"]
    else:
        assert "CHECKING COPY" in turn["instructions"]
        assert "never the raw evidence locators it withholds" in turn["instructions"]
        assert "not an OS sandbox" in turn["instructions"]
