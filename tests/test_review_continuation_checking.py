"""A writable reviewer's checking copy survives a continuable stop and moves to its explicit successor.

A stop the engine can continue keeps the one-shot copy after its receipt; the explicit choice hands
the SAME path to the successor operation (its edits, deletions and evidence intact, never rebuilt from
the frozen subject), and nothing the superseded operation still runs (a late capture, the prune pass)
can release it. A copy already released is a named gap, never a restored copy. Copies, receipts,
markers and locks are real; custody is stubbed for the prune as in its own suite.
"""
from __future__ import annotations

import time
from pathlib import Path

from ouroboros import review_checking as rc
from ouroboros import review_checking_prune as rp
from ouroboros.review_continuation import checking_copy_holder, transfer_checking_copy
from tests._review_session_route_shared import _owned_gateway_uses_each_test_transport  # noqa: F401 - no real engine
from tests.test_review_checking_prune import LATER, _custody
from tests.test_review_checkout_lifecycle import EDIT, TASK, _frozen, _seat, _thread, _work

TERMINAL = {"summary": {"state": "failed"}}
SUCCEEDED = {"summary": {"state": "succeeded"}}


def test_a_continuable_stop_keeps_the_copy_and_its_successor_works_and_releases_it_in_place(tmp_path, monkeypatch):
    root, source = _frozen(tmp_path)
    area = _seat(root, source, "op-1")
    (area / "check.py").unlink()  # a deletion is part of the reviewer's work too
    (area / "found.txt").write_bytes(b"first run")
    receipt = rc.terminal_seat_evidence(root, TASK, "op-1", TERMINAL, hold=True)
    assert receipt["status"] in ("retained", "partial") and area.is_dir()  # receipted, kept for a successor
    assert checking_copy_holder(root, TASK, "op-1") == (True, "")
    _custody(monkeypatch)  # no run is live in the copy: only the hold keeps it while the task lives
    rp.prune_review_checkouts(root, {TASK})
    assert area.is_dir()

    transfer_checking_copy(root, TASK, "op-1", "op-2")
    transfer_checking_copy(root, TASK, "op-1", "op-2")  # idempotent projection
    assert checking_copy_holder(root, TASK, "op-2") == (True, "")
    assert checking_copy_holder(root, TASK, "op-1")[0] is False  # superseded: it no longer holds the copy
    assert not (area / "check.py").exists() and (area / "found.txt").read_bytes() == b"first run"
    late = rc.terminal_seat_evidence(root, TASK, "op-1", TERMINAL)  # a superseded late capture
    assert late == receipt and area.is_dir()
    rp.prune_review_checkouts(root, set(), now=time.time() + LATER)  # the task ended; the holder ran nothing yet
    assert area.is_dir(), "a copy handed to an unsent successor is never released by its predecessor's receipt"

    (area / "found.txt").write_bytes(b"successor finished")
    final = rc.terminal_seat_evidence(root, TASK, "op-2", SUCCEEDED)
    assert final["status"] in ("retained", "partial") and not area.exists()  # captured from the same path, then released
    retained = Path(final["evidence_dir"]) / "files" / "found.txt"
    assert retained.read_bytes() == b"successor finished"


def test_a_successor_that_stops_again_keeps_the_copy_for_the_next_choice(tmp_path, monkeypatch):
    root, source = _frozen(tmp_path)
    area = _seat(root, source, "op-1")
    rc.terminal_seat_evidence(root, TASK, "op-1", TERMINAL, hold=True)
    transfer_checking_copy(root, TASK, "op-1", "op-2")
    rc.terminal_seat_evidence(root, TASK, "op-2", TERMINAL, hold=True)
    assert area.is_dir() and checking_copy_holder(root, TASK, "op-2") == (True, "")
    _custody(monkeypatch)
    rp.prune_review_checkouts(root, {TASK})
    assert area.is_dir()
    transfer_checking_copy(root, TASK, "op-2", "op-3")
    assert checking_copy_holder(root, TASK, "op-3") == (True, "") and checking_copy_holder(root, TASK, "op-2")[0] is False


def test_a_released_copy_is_a_named_gap_never_a_rebuilt_one(tmp_path):
    root, source = _frozen(tmp_path)
    area = _seat(root, source, "op-1")
    rc.terminal_seat_evidence(root, TASK, "op-1", TERMINAL)  # an ordinary stop: released after its receipt
    assert not area.exists()
    held, why = checking_copy_holder(root, TASK, "op-1")
    assert held is False and "released" in why
    assert checking_copy_holder(root, TASK, "op-unknown") == (False, "no checking copy was recorded for this operation")


def test_a_thread_seat_moves_to_its_successor_turn_without_rematerializing(tmp_path):
    root, subject = tmp_path / "data", tmp_path / "subject"
    subject.mkdir()
    (subject / "check.py").write_bytes(b"# v1\n")
    from tests.test_review_checkout_lifecycle import directory_source

    source = directory_source(subject, root)
    area = _thread(root, source, "turn-1")
    _work(area)
    rc.terminal_seat_evidence(root, TASK, "turn-1", TERMINAL)  # a thread copy stays after its receipt
    transfer_checking_copy(root, TASK, "turn-1", "turn-2")
    assert (area / "check.py").read_bytes() == EDIT  # the stopped turn's work, not the frozen subject again
    assert checking_copy_holder(root, TASK, "turn-2") == (True, "")
    (area / "check.py").write_bytes(b"# continued\n")
    receipt = rc.terminal_seat_evidence(root, TASK, "turn-2", SUCCEEDED)
    assert receipt["status"] in ("retained", "partial") and area.is_dir()
    assert (Path(receipt["evidence_dir"]) / "files" / "check.py").read_bytes() == b"# continued\n"


def test_a_writable_successor_runs_in_the_held_copy_its_predecessor_left(tmp_path, monkeypatch):
    """Transport and projection together: the stopped run's copy stays, the successor's start names the SAME
    copy, finds the earlier work there, and its own terminal releases the copy after its receipt."""
    import copy as copy_module

    import pytest

    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody import invocation_record
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.observability import write_blob
    from ouroboros.review_continuation import _start_requested
    from ouroboros.review_execution import run_delegated_review_session
    from tests._review_session_route_shared import FakeGateway, _terminal_detail
    from tests.test_review_session_rights import _git, _invocation

    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (("init", "-q"), ("config", "user.email", "a@b"), ("config", "user.name", "a")):
        _git(repo, *args)
    (repo / "f.txt").write_text("v0\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    (repo / "f.txt").write_text("staged\n")
    _git(repo, "add", "f.txt")
    FakeGateway.reset()
    FakeGateway.run_dir = tmp_path / "review-run"
    monkeypatch.setattr("ouroboros.gateways.claudexor.ClaudexorGateway", FakeGateway)
    monkeypatch.setenv("OUROBOROS_REVIEW_SESSION_ROUTE", "fake-review=fake-small:low")
    custody._CUSTODY.clear()
    FakeGateway.catalog_entry["accessProfilesSupported"] = ["readonly", "workspace_write", "full"]
    original_init = FakeGateway.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.engine_version = "3.22.0"
        self.ensure_full_access = lambda root: {"allowFullAccess": True}
    monkeypatch.setattr(FakeGateway, "__init__", init)
    drive = tmp_path / "drive"
    source = rc.freeze_checking_source(repo, data_root=drive, subject=rc.SUBJECT_INDEX)
    area = rc.seat_dir(drive, "task-1", "op-1") / "area.noindex"
    runs = {}

    def start(self, body, *, idempotency_key=""):
        self.start_requests.append(copy_module.deepcopy(body))
        return {"runId": runs.setdefault(idempotency_key, f"run-{len(runs) + 1}")}

    def detail(self, run_id, **_kw):
        if run_id == "run-1":
            (area / "first.txt").write_text("first run's work")
            (area / "f.txt").unlink()
            result = _terminal_detail("partial", state="failed")
            result["summary"].update(failure={"code": "vendor_limit", "safeMessage": "limit"},
                                     resumable={"cause": "vendor_limit", "carriers": ["native"]})
            return result
        assert (area / "first.txt").read_text() == "first run's work" and not (area / "f.txt").exists()
        (area / "second.txt").write_text("successor's work")
        return _terminal_detail("[]", conformance="passed")
    monkeypatch.setattr(FakeGateway, "start_run", start)
    monkeypatch.setattr(FakeGateway, "get_run", detail)

    keys = []
    with pytest.raises(ClaudexorUnavailable):
        run_delegated_review_session(prompt="review", root=str(repo), custody_drive=drive, invocation=_invocation(
            access="workspace_write", checking_source=source, pending_invocation_checkpoint=keys.append))
    assert area.is_dir() and rc._receipt(drive, "task-1", "op-1")["status"] in ("retained", "partial")
    original = invocation_record(drive, keys[0])["request"]
    successor = {**original, "prompt": "continue", "continueFrom": "run-1"}
    decision = {"operation_id": "op-2", "invocation_id": "token-2", "surface": "multi_model_review", "thread_id": "",
                "continues_run": "run-1", "predecessor_invocation_id": keys[0], "origin_invocation_id": keys[0],
                "request_ref": write_blob(drive, successor, kind="json"), "supersedes": "op-1"}
    transfer_checking_copy(drive, "task-1", "op-1", "op-2")
    assert _start_requested(drive, "task-1", "t1", decision) is True
    recorded = invocation_record(drive, "token-2")
    assert recorded["execution_root"] == str(area) and recorded["request"]["execution"]["workspaceRoot"] == str(area)
    custody._CUSTODY.clear()
    facts = run_delegated_review_session(prompt="continue", root=str(repo), custody_drive=drive, invocation=_invocation(
        access="workspace_write", checking_source=source, operation_id="op-2",
        retry_state={"pending_invocation_id": "token-2"}, reconcile_only=True))
    assert facts["text"] == "[]" and not area.exists()  # captured from the same path, then released
    receipt = facts["checking_evidence"]
    assert {"first.txt", "second.txt", "f.txt"} <= {line.split("\t")[-1] for line in receipt["changed"]}
    patch = (Path(receipt["evidence_dir"]) / "delta.patch").read_text()
    assert "successor's work" in patch and "first run's work" in patch  # both runs' work, one copy
