"""Read failures preserve known Wrap up facts and disclose missing text, not absence."""
import io

import pytest

from ouroboros import cancel_intents, delegate_terminal, owner_stop_receipt as receipt, task_results
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_owner_stop_retained_work import (
    CHILD_FIELD, _capture, _child, _collect, _document, _wrap_up, _write_reciprocal_retry_root,
)


@pytest.mark.parametrize("owner", ["parent1", "kid"])
@pytest.mark.parametrize("fail_once", [True, False])
def test_recapture_snapshot_failure_keeps_verified_run_facts(tmp_path, monkeypatch, owner, fail_once):
    write_task_result(tmp_path, "parent1", "running")
    _child(tmp_path, result="child answer")
    _collect(tmp_path, monkeypatch, tid=owner, money=True)
    _, stop = _wrap_up(tmp_path)
    first = _capture(tmp_path, "parent1", stop)[CHILD_FIELD]

    def run(row):
        return (row["runs"] if owner == "parent1" else row["children"][0]["custody"]["terminal_runs"])[0]

    before = run(first)
    original = delegate_terminal.custody_audit_snapshot
    calls = 0

    def unavailable(root, *, task_id=""):
        nonlocal calls
        if task_id == owner:
            calls += 1
            if not fail_once or calls == 1:
                raise OSError("snapshot unavailable")
        return original(root, task_id=task_id)

    monkeypatch.setattr(delegate_terminal, "custody_audit_snapshot", unavailable)
    after = run(_capture(tmp_path, "parent1", stop)[CHILD_FIELD])
    for key in ("accounting", "patch", "answer_preview", "output_state", "engine_continuity"):
        assert after.get(key) == before[key], key
    assert after.get("gaps"), "retained facts must be marked last-known after a failed read"


@pytest.mark.parametrize("with_child", [False, True])
def test_root_audit_failure_never_claims_no_saved_text(tmp_path, monkeypatch, with_child):
    write_task_result(tmp_path, "parent1", "running")
    if with_child:
        _child(tmp_path, result="")
    _, stop = _wrap_up(tmp_path)
    original = delegate_terminal.custody_audit_snapshot

    def unavailable(root, *, task_id=""):
        if task_id == "parent1":
            raise OSError("root custody unavailable")
        return original(root, task_id=task_id)

    monkeypatch.setattr(delegate_terminal, "custody_audit_snapshot", unavailable)
    _capture(tmp_path, "parent1", stop)
    stored = load_task_result(tmp_path, "parent1")
    assert receipt.wrap_up_document_view(stored), "an unknown root must not hide the saved-work door"
    text = _document(tmp_path)
    assert "This work kept no saved text." not in text
    assert "could not" in text


@pytest.mark.parametrize("missing", [None, {}, {"result": ""}])
def test_download_names_a_recorded_child_answer_that_is_now_unavailable(tmp_path, monkeypatch, missing):
    write_task_result(tmp_path, "parent1", "running")
    _child(tmp_path, result="recorded answer", terminal_origin="model_final")
    _, stop = _wrap_up(tmp_path)
    _capture(tmp_path, "parent1", stop)
    root_result = load_task_result(tmp_path, "parent1")
    original = task_results.load_task_result
    monkeypatch.setattr(task_results, "load_task_result", lambda root, tid, *a, **k:
                        missing if tid == "kid" else original(root, tid, *a, **k))
    sink = io.BytesIO()
    receipt.write_wrap_up_document(tmp_path, root_result, sink)
    text = sink.getvalue().decode()
    assert "This work kept no saved text." not in text
    assert "stored result could not be read" in text or "recorded text is no longer" in text


def test_predecessor_publication_failure_keeps_known_refs_and_names_gap(tmp_path, monkeypatch):
    _write_reciprocal_retry_root(tmp_path, "L", "R")
    _collect(tmp_path, monkeypatch, tid="L", rid="run-pred", text="predecessor output")
    original = delegate_terminal.record_terminal_reconciliation
    original(tmp_path, "L", delegate_terminal.terminal_custody_disclosure(tmp_path, "L"))

    def unavailable(root, tid, audit, *a, **k):
        if tid == "L":
            raise OSError("publication unavailable")
        return original(root, tid, audit, *a, **k)

    monkeypatch.setattr(delegate_terminal, "record_terminal_reconciliation", unavailable)
    audit = receipt.capture_receipt(tmp_path, "R", {}, stop=("L", "ci_saved"))
    saved = audit[CHILD_FIELD]
    assert [row["run_id"] for row in saved["runs"]] == ["run-pred"]
    assert saved.get("gap")
    assert "predecessor output" in _document(tmp_path, "R")


def test_unreadable_retry_resolution_is_not_a_raw_target_identity(tmp_path, monkeypatch):
    _write_reciprocal_retry_root(tmp_path, "L", "R")
    def unavailable(*_a, **_k):
        raise OSError("retry chain unavailable")
    monkeypatch.setattr(cancel_intents, "_validated_single_cancel_target", unavailable)
    with pytest.raises(OSError, match="retry chain unavailable"):
        receipt.capture_receipt(tmp_path, "L", {"R": "cancelled"}, stop=("L", "ci_saved"))
    assert CHILD_FIELD not in load_task_result(tmp_path, "L").get("delegate_terminal_reconciliation", {})


def test_multiline_notice_is_one_compact_bullet_and_full_original_text(tmp_path):
    write_task_result(tmp_path, "parent1", "running")
    notice = "First paragraph.\n\nSecond paragraph: " + "long note " * 30
    _child(tmp_path, result="", terminal_host_notice=notice)
    _, stop = _wrap_up(tmp_path)
    audit = _capture(tmp_path, "parent1", stop)
    compact = receipt.receipt_text(audit, owner_row=True)
    assert all(line.startswith("- ") for line in compact.splitlines()[1:]), compact
    full = _document(tmp_path)
    assert "> First paragraph.\n>\n> Second paragraph:" in full
