"""Cold collection retains a terminal reviewer's real checking-copy evidence."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from ouroboros import delegate_custody as custody, review_checking as checking
from ouroboros.review_dispatch import review_operation_binding
from ouroboros.review_records import ReviewRequest, ReviewSlot
from tests.test_review_checkout_lifecycle import ABSENT, NO_CENSUS, SETTLED, directory_source
from tests.test_review_operation_collection import (
    PASS, SURFACE, TASK, _collect as _collect_seat, _ctx, _observe, _session_run, _started, _verdict,
    forbid_effects,  # noqa: F401
)

OPERATION = "op-session"
INVOCATION = f"inv-{OPERATION}"
RUN = "run-paid"
COVERAGE = b'<?xml version="1.0"?><coverage lines-covered="3"/>\n'
REVIEWER_EDIT = b"# reviewer reproduction\nassert 2 + 2 == 4\n"
FAIL = json.dumps([{"item": "intent", "verdict": "FAIL", "severity": "critical",
                    "reason": "Independent review rejects the answer: check.py needs repair"}])


@pytest.fixture
def checking_run(tmp_path):
    """A dispatched full-access seat, with its real retained subject and copy."""
    root, subject = tmp_path / "data", tmp_path / "subject"
    subject.mkdir()
    (subject / "check.py").write_bytes(b"# original candidate\n")
    source = directory_source(subject, root)  # a copied subject: a skill seat's retained view
    area = Path(checking.prepare_seat_area(root, TASK, OPERATION, source))
    (area / "check.py").write_bytes(REVIEWER_EDIT)
    (area / checking.EVIDENCE_DIRNAME).mkdir()
    (area / checking.EVIDENCE_DIRNAME / "coverage.xml").write_bytes(COVERAGE)
    run = _session_run()
    run["request"]["checking_source"] = source
    run["slot_roster"][0]["session_access"] = "full"
    from ouroboros.review_dispatch import review_reconciliation_identity
    request, slot = ReviewRequest(**run["request"]), ReviewSlot(**run["slot_roster"][0])
    request.reconciliation_identity = review_reconciliation_identity(request, [slot], root_task_id=TASK)
    run["request"] = dataclasses.asdict(request)
    binding = review_operation_binding(request, slot, OPERATION)
    run["actors"][0].update(
        pending_invocation_id=INVOCATION,
        recovery_binding={**binding, "pending_invocation_id": INVOCATION})
    from ouroboros.observability import persist_call
    persist_call(root, task_id=TASK, call_id=OPERATION + "_prompt", call_type="review_prompt",
                 payload={"request": run["request"], "slot": run["slot_roster"][0]},
                 manifest={"review_operation_binding": binding})
    return root, subject, area, run


def _terminal(state="succeeded", *, scope=ABSENT, text=PASS):
    summary = {"state": state, **({} if scope is ABSENT else {"processScope": scope})}
    return {"summary": summary, "primaryOutput": {"text": text, "truncated": False}}


def _collect(root, run):
    return _collect_seat(run, root, _ctx(root))


def _assert_receipt(actor, area):
    assert "checking_evidence" in actor["usage"], actor.get("error")
    receipt = actor["usage"]["checking_evidence"]
    assert receipt["status"] == "retained" and receipt["gaps"] == []
    assert not area.exists(), "a terminal run's copy goes once its receipt is written"
    assert not {"area", "refresh_safe", "retention_reason", "release_basis", "process_scope"} & set(receipt)
    evidence = Path(receipt["evidence_dir"])
    assert (evidence / "files" / "check.py").read_bytes() == REVIEWER_EDIT
    coverage = evidence / "files" / checking.EVIDENCE_DIRNAME / "coverage.xml"
    assert coverage.read_bytes() == COVERAGE
    # A reviewer's copy keeps its source's mode, so the receipt names the mode the retained file
    # carries (a writable file reads 0o666 on Windows), never a POSIX default.
    assert receipt["named_evidence"] == [{
        "path": f"{checking.EVIDENCE_DIRNAME}/coverage.xml", "size": len(COVERAGE),
        "sha256": hashlib.sha256(COVERAGE).hexdigest(), "mode": stat.S_IMODE(coverage.stat().st_mode),
    }]
    assert "check.py" in receipt["changed"]
    assert json.loads((evidence / "summary.json").read_text(encoding="utf-8")) == receipt
    assert actor["operation_id"] == OPERATION
    assert actor["usage"]["invocation_id"] == INVOCATION
    assert not actor["usage"].get("pending_invocation_id"), "a terminal token must not create a new live wait"
    assert actor["recovery_binding"]["pending_invocation_id"] == INVOCATION
    assert actor["operation_state"] == "late_settled"
    return receipt


@pytest.mark.parametrize("state", ["succeeded", "failed", "cancelled", "interrupted"])
def test_attach_only_terminal_collection_captures_edits_and_named_evidence(
        checking_run, monkeypatch, forbid_effects, state):  # noqa: F811
    root, subject, area, run = checking_run
    calls = _observe(monkeypatch, _terminal(state))
    _started(root, run_id=RUN)
    actor, fact = _collect(root, run)
    _assert_receipt(actor, area)
    assert actor["recovery_binding"] == run["actors"][0]["recovery_binding"]
    assert fact == "collected"
    assert actor["operation_state"] not in {"pending_dispatch", "in_flight"}
    assert (subject / "check.py").read_bytes() == b"# original candidate\n"
    assert not (subject / checking.EVIDENCE_DIRNAME).exists()
    assert calls == [("handshake",), ("get_run", RUN), ("close",)]
    if state == "succeeded":
        assert _verdict(actor) == "PASS"
    else:
        assert actor["status"] == "error" and state in actor["error"]
        assert not actor.get("semantic_verdict")


def test_repeat_attach_collection_reuses_the_retained_receipt_without_recreating_copy(
        checking_run, monkeypatch, forbid_effects):  # noqa: F811
    root, _subject, area, run = checking_run
    calls = _observe(monkeypatch, _terminal())
    _started(root, run_id=RUN)
    first, _fact = _collect(root, run)
    receipt = _assert_receipt(first, area)
    summary = Path(receipt["evidence_dir"], "summary.json")
    summary_bytes, summary_mtime = summary.read_bytes(), summary.stat().st_mtime_ns

    def forbidden(*_args, **_kwargs):
        pytest.fail("repeat collection recomputed or rematerialized the terminal copy")

    monkeypatch.setattr(checking, "_directory_delta", forbidden)
    monkeypatch.setattr(checking, "prepare_seat_area", forbidden)
    again, _fact = _collect(root, run)  # an old pending checkpoint observes the same paid run
    assert _assert_receipt(again, area) == receipt
    assert summary.read_bytes() == summary_bytes and summary.stat().st_mtime_ns == summary_mtime
    assert calls == [("handshake",), ("get_run", RUN), ("close",)]  # S3: the second read uses producer CAS


@pytest.mark.parametrize("verdict", ["PASS", "FAIL"])
@pytest.mark.parametrize("scope", [ABSENT, None, SETTLED, NO_CENSUS, {**NO_CENSUS, "survivors": 1}],
                         ids=["absent", "null", "settled", "no-census", "survivor"])
def test_any_process_scope_releases_the_copy_and_collects_the_paid_verdict(
        checking_run, monkeypatch, forbid_effects, scope, verdict):  # noqa: F811
    """The census the engine reports does not decide anything here (owner 1A, 2026-10-12):
    the evidence is kept, the copy goes, and the paid verdict arrives from one observation."""
    root, subject, area, run = checking_run
    calls = _observe(monkeypatch, _terminal(scope=scope, text={"PASS": PASS, "FAIL": FAIL}[verdict]))
    _started(root, run_id=RUN)
    actor, fact = _collect(root, run)
    _assert_receipt(actor, area)
    assert _verdict(actor) == verdict
    assert fact == "collected" and actor["operation_state"] not in {"pending_dispatch", "in_flight"}
    assert calls == [("handshake",), ("get_run", RUN), ("close",)], "observed once, nothing bought again"
    assert (subject / "check.py").read_bytes() == b"# original candidate\n"


def test_a_receipt_captured_before_an_engine_upgrade_is_reused_as_stored(
        checking_run, monkeypatch, forbid_effects):  # noqa: F811
    root, _subject, area, run = checking_run
    _observe(monkeypatch, _terminal())  # an older engine: released after complete capture
    _started(root, run_id=RUN)
    receipt = _assert_receipt(_collect(root, run)[0], area)
    summary = Path(receipt["evidence_dir"], "summary.json")
    summary_bytes, summary_mtime = summary.read_bytes(), summary.stat().st_mtime_ns

    def forbidden(*_args, **_kwargs):
        pytest.fail("a stored receipt was recaptured or its copy rematerialized")

    monkeypatch.setattr(checking, "_directory_delta", forbidden)
    monkeypatch.setattr(checking, "prepare_seat_area", forbidden)
    calls = _observe(monkeypatch, _terminal(scope=None))  # the upgraded engine reads the same run as null
    again = _collect(root, run)[0]
    assert _assert_receipt(again, area) == receipt and _verdict(again) == "PASS"
    assert summary.read_bytes() == summary_bytes and summary.stat().st_mtime_ns == summary_mtime
    assert calls == []  # S3: a complete retained producer needs no new engine observation


@pytest.mark.parametrize("gap", ["special_entry", "capture_error"])
def test_attach_collection_names_capture_gaps_and_keeps_only_a_failed_copy(
        checking_run, monkeypatch, forbid_effects, gap):  # noqa: F811
    root, _subject, area, run = checking_run
    _observe(monkeypatch, _terminal())
    _started(root, run_id=RUN)
    if gap == "special_entry":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO output requires POSIX")
        os.mkfifo(area / "pipe")  # a reviewer output no capture can retain
    else:
        def unavailable(*_args, **_kwargs):
            raise OSError("fixture evidence read failed")
        monkeypatch.setattr(checking, "_directory_delta", unavailable)
    actor, _fact = _collect(root, run)
    receipt = actor["usage"]["checking_evidence"]
    assert _verdict(actor) == "PASS", "evidence capture does not erase the paid answer"
    assert actor["recovery_binding"] == run["actors"][0]["recovery_binding"]
    if gap == "special_entry":  # a named gap is a receipt: the copy goes
        assert receipt["status"] == "partial" and not area.exists()
        assert receipt["gaps"] == [{"path": "pipe", "reason": "unsupported_entry_type"}]
    else:  # nothing goes without a receipt of what was kept
        assert receipt["status"] == "capture_failed" and receipt["area"] == str(area)
        assert "fixture evidence read failed" in receipt["error"]
        assert (area / "check.py").read_bytes() == REVIEWER_EDIT
        assert (area / checking.EVIDENCE_DIRNAME / "coverage.xml").read_bytes() == COVERAGE


@pytest.mark.parametrize("state", ["running", "daemon_unavailable", "unbound"])
def test_unfinished_or_unavailable_invocation_never_captures_or_releases(
        checking_run, monkeypatch, forbid_effects, state):  # noqa: F811
    root, _subject, area, run = checking_run
    # An answer can precede adapter completion; only the final daemon summary
    # supplies the normal-success capture/release boundary.
    calls = _observe(monkeypatch, {"summary": {"state": "running"},
                                   "primaryOutput": {"text": PASS, "truncated": False}})
    _started(root, run_id="" if state == "unbound" else RUN)

    def forbidden(*_args, **_kwargs):
        pytest.fail("nonterminal or unavailable invocation reached evidence capture")

    monkeypatch.setattr(checking, "capture_seat_evidence", forbidden)
    if state == "daemon_unavailable":
        def unavailable():
            raise ConnectionError("fixture daemon unavailable")
        monkeypatch.setattr("ouroboros.claudexor_daemon.read_owned_gateway", unavailable)
    actor, fact = _collect(root, run)
    assert fact == ("unavailable" if state == "unbound" else "deferred")
    assert actor["late_result_pending"]
    assert actor["recovery_binding"] == run["actors"][0]["recovery_binding"]
    assert area.exists() and (area / "check.py").read_bytes() == REVIEWER_EDIT
    assert not (area.parent / "evidence").exists()
    if state != "running":
        assert calls == []


@pytest.mark.parametrize("foreign", ["task_id", "root_task_id", "operation_id", "slot_id", "surface"])
def test_foreign_invocation_binding_cannot_collect_or_release_checking_copy(
        checking_run, monkeypatch, forbid_effects, foreign):  # noqa: F811
    root, _subject, area, run = checking_run
    calls = _observe(monkeypatch, _terminal())
    requested = {
        "invocation_id": INVOCATION, "task_id": TASK, "root_task_id": TASK,
        "operation_id": OPERATION, "slot_id": "s", "surface": SURFACE,
        "request": {"prompt": "review", "harnesses": ["codex"]},
    }
    requested[foreign] = "foreign-value"
    custody.emit(root, custody.START_REQUESTED, requested)
    custody.emit(root, custody.STARTED, {"run_id": RUN, "invocation_id": INVOCATION, "task_id": TASK})

    def forbidden(*_args, **_kwargs):
        pytest.fail("foreign invocation reached evidence capture")

    monkeypatch.setattr(checking, "capture_seat_evidence", forbidden)
    before = copy.deepcopy(run["actors"][0]["recovery_binding"])
    actor, fact = _collect(root, run)
    assert fact == "unavailable"
    assert actor["operation_state"] == "custody_lost"
    assert actor["late_result_pending"]
    assert actor["recovery_binding"] == before
    assert calls == [] and area.exists()
    assert not (area.parent / "evidence").exists()


def test_capture_exception_does_not_hide_the_paid_verdict_or_release_the_copy(
        checking_run, monkeypatch, forbid_effects):  # noqa: F811
    root, _subject, area, run = checking_run
    _observe(monkeypatch, _terminal())
    _started(root, run_id=RUN)
    captures = []

    def failed_capture(data, task, operation):
        captures.append((data, task, operation))
        raise OSError("capture unavailable")

    monkeypatch.setattr(checking, "capture_seat_evidence", failed_capture)
    actor, fact = _collect(root, run)
    assert captures == [(root, TASK, OPERATION)]
    assert _verdict(actor) == "PASS"
    assert fact == "collected"
    assert actor["usage"]["checking_evidence"]["status"] == "capture_failed"
    assert "capture unavailable" in actor["usage"]["checking_evidence"]["error"]
    assert area.exists() and (area / checking.EVIDENCE_DIRNAME / "coverage.xml").read_bytes() == COVERAGE
