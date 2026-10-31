"""Every settlement writer merges against the locked current per-seat record."""
import concurrent.futures
import copy
import threading

import pytest

from ouroboros import review_ledger as ledger
from tests.test_review_ledger import _facts, _raw, BOTH


def _record(root):
    raws = [_raw(seat, seat, "in_flight", parts=BOTH, operation_id=seat,
                 operation_state="in_flight", late_result_pending=True) for seat in ("a", "b")]
    return ledger.write_record(root, ledger.build_commit_gate_record(_facts(raws), drive_root=root))


def test_concurrent_collectors_keep_both_answers_cost_and_author_decision(tmp_path, monkeypatch):
    record = _record(tmp_path)
    ledger.note_author_decision(tmp_path, record["record_id"], {"disposition": "deferred", "rationale": "wait"})
    original = ledger.revise_record
    barrier = threading.Barrier(2)

    def after_snapshot(*args, **kwargs):
        barrier.wait(timeout=10)  # both callers read the old pending revision
        return original(*args, **kwargs)
    monkeypatch.setattr(ledger, "revise_record", after_snapshot)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(ledger.settle_from_collected, tmp_path, record["record_id"],
                    [_raw(seat, seat, parts=BOTH, operation_id=seat)]) for seat in ("a", "b")]
        for future in futures:
            future.result()
    settled = ledger.load_record(tmp_path, record["record_id"])
    assert settled["state"] == "settled" and settled["verdict"]["aggregate"] == "PASS"
    assert settled["cost"]["usd"] == .02
    assert settled["author_decision"]["disposition"] == "deferred"
    assert {row["seat_id"] for row in settled["rows"] if row["status"] == "responded"} == {"a", "b"}


def test_stale_writer_cannot_replace_answer_or_provenance(tmp_path):
    original = _record(tmp_path)
    fresh = ledger.settle_from_collected(tmp_path, original["record_id"], [_raw("a", "a", parts=BOTH, operation_id="a")])
    stale = copy.deepcopy(original)
    stale["subject"]["tree_sha"] = "different"
    merged = ledger.revise_record(tmp_path, original["record_id"], lambda now: ledger.settle_pending_payload(now, stale))
    assert merged["rows"][0]["status"] == "responded"
    assert merged["subject"] == fresh["subject"]
    assert merged["rows"][1]["status"] == "in_flight"
    # A genuinely new answer to the still-open seat is accepted.
    settled = ledger.settle_from_collected(tmp_path, original["record_id"], [_raw("b", "b", parts=BOTH, operation_id="b")])
    assert settled["verdict"]["aggregate"] == "PASS"


@pytest.mark.parametrize("writer", ["collected", "pending_payload"])
@pytest.mark.parametrize("status,role", [("responded", "response"), ("error", "failure")])
def test_settlement_replaces_current_sources_and_keeps_historical_originals(tmp_path, writer, status, role):
    from ouroboros.review_projection import ledger_record_panel
    from tests.test_task_review_sources import client_for, source_url

    note = "Timeout at dispatch; physical review operation remains in flight."
    raw = _raw("sol", "m", "pending", parts=BOTH, operation_id="op", operation_state="in_flight",
               raw_text=note, failure_code="logical_timeout", reported_cause=note)
    record = ledger.write_record(tmp_path, ledger.build_commit_gate_record(_facts([raw]), drive_root=tmp_path))
    assert not ledger_record_panel(record)["actors"][0]["sources"], "pending note is not a failure or an answer"
    # Also repair a record produced by the old implementation, preserving its immutable URL.
    old = ledger.retain_text_source(tmp_path, "task-1", record_id=record["record_id"], seat_id="sol",
                                    role="failure", part="change,coupling", text=note)
    record = ledger.revise_record(tmp_path, record["record_id"], lambda r: {
        **r, "rows": [{**r["rows"][0], "source_refs": [*r["rows"][0]["source_refs"], old]}]})
    body = "Whole late answer\nEND" if status == "responded" else "Actual terminal refusal\nEND"
    late = _raw("sol", "m", status, parts=BOTH, operation_id="op", operation_state="settled", raw_text=body)
    if writer == "collected":
        settled = ledger.settle_from_collected(tmp_path, record["record_id"], [late])
    else:
        fresh = ledger.build_commit_gate_record(_facts([late]), record_id=record["record_id"], drive_root=tmp_path).to_dict()
        settled = ledger.revise_record(tmp_path, record["record_id"], lambda r: ledger.settle_pending_payload(r, fresh))
    actor = ledger_record_panel(settled)["actors"][0]
    assert len(actor["sources"]) == 1 and actor["sources"][0]["role"] == role
    assert note not in actor["reason"] and actor.get("failure_code") != "logical_timeout"
    assert (settled["verdict"]["aggregate"] == "PASS") == (status == "responded")
    current = next(s for s in settled["rows"][0]["source_refs"] if s["role"] == role)
    assert old in settled["rows"][0]["historical_source_refs"]
    assert ledger.record_sources_resolvable(tmp_path, settled)
    with client_for(tmp_path) as client:
        assert client.get(source_url("task-1", settled, current, role=role)).content == body.encode()
        assert client.get(source_url("task-1", settled, old, role="failure")).content == note.encode()


def test_late_owner_settlement_keeps_the_collected_reply_call_refs(tmp_path):
    """A left wave's reply collected by review_pause keeps its own call refs beside the dispatch prompt."""
    from ouroboros import review_pause
    from ouroboros.review_projection import ledger_record_panel
    from tests.test_review_seat_outcome import _body

    note = "Timeout at dispatch; physical review operation remains in flight."
    dispatch_call = {"call_id": "dispatch-error", "manifest_ref": {"path": "calls/dispatch-error.json", "sha256": "1" * 64}}
    pending = _raw("sol", "m", "timeout", parts=BOTH, operation_id="op", operation_state="in_flight",
                   raw_text=note, response_ref=dispatch_call)
    record = ledger.write_record(tmp_path, ledger.build_commit_gate_record(_facts([pending]), drive_root=tmp_path))
    [prompt] = [s for s in record["rows"][0]["source_refs"] if s["role"] == "prompt"]
    late_call = {"call_id": "late-reply", "manifest_ref": {"path": "calls/late-reply.json", "sha256": "2" * 64}}
    seat = {"slot_id": "sol", "model": "m", "parts": list(BOTH), "operation_id": "op", "operation_state": "settled",
            "status": "responded", "raw_text": _body(), "prompt_ref": {"call_id": "late-prompt"}, "response_ref": late_call}
    settled = review_pause._settle_left_record({"drive_root": str(tmp_path), "record_id": record["record_id"]},
                                               [seat], final=True)
    assert settled["state"] == "settled" and settled["verdict"]["aggregate"] == "PASS"
    row = settled["rows"][0]
    current = {s["role"]: s for s in row["source_refs"]}
    assert current["prompt"] == prompt and current["response"]["status"] == "retained"
    assert current["observability_response"]["ref"] == late_call
    assert ledger_record_panel(settled)["actors"][0]["response_ref"] == {"call_id": "late-reply", "manifest_sha256": "2" * 64}
    historical = row["historical_source_refs"]
    assert [s["role"] for s in historical if s.get("ref", {}) == dispatch_call] == ["observability_response"]
    assert [s for s in historical if s["role"] == "pending" and s["chars"] == len(note)]


def test_additional_pending_does_not_hold_counted_pass(tmp_path):
    record = _record(tmp_path)
    ledger.revise_record(tmp_path, record["record_id"],
                         lambda row: {**row, "rows": [row["rows"][0], {**row["rows"][1], "additional": True}]})
    settled = ledger.settle_from_collected(tmp_path, record["record_id"], [_raw("a", "a", parts=BOTH, operation_id="a")])
    assert settled["state"] == "pending" and settled["verdict"]["aggregate"] == "PASS"
    # The same open seat counted in a fresh record keeps the verdict pending.
    other = _record(tmp_path)
    pending = ledger.settle_from_collected(tmp_path, other["record_id"], [_raw("a", "a", parts=BOTH, operation_id="a")])
    assert pending["verdict"]["aggregate"] != "PASS"


@pytest.mark.parametrize("operation", ["a", "other-producer"])
def test_collected_answer_must_belong_to_the_recorded_operation(tmp_path, operation):
    record = _record(tmp_path)
    settled = ledger.settle_from_collected(tmp_path, record["record_id"],
                                           [_raw("a", "a", parts=BOTH, operation_id=operation)])
    assert settled["rows"][0]["operation_id"] == "a"
    assert (settled["rows"][0]["status"] == "responded") == (operation == "a")


def test_findings_merge_by_key_and_a_fresh_gate_block_is_kept(tmp_path):
    original = _record(tmp_path)
    complete = ledger.settle_from_collected(tmp_path, original["record_id"],
                                             [_raw(seat, seat, parts=BOTH) for seat in ("a", "b")])
    original["verdict"]["advisory_findings"] = [{"seat_id": "a", "part": "change", "item": "issue", "obligation_id": "same", "reason": "old"}]
    complete["verdict"].update(reason="gate_block:binding_mismatch", aggregate="NOT_PERFORMED",
        advisory_findings=[{"seat_id": "a", "part": "change", "item": "issue", "obligation_id": "same", "reason": "new"}])
    merged = ledger.settle_pending_payload(original, complete)
    assert len(merged["verdict"]["advisory_findings"]) == 1
    assert merged["verdict"]["reason"] == "gate_block:binding_mismatch"
    complete["verdict"]["reason"] = "pass"
    assert ledger.settle_pending_payload(original, complete)["verdict"]["aggregate"] == "PASS"


@pytest.mark.parametrize("writer", ["collected", "pending_payload"])
def test_distinct_findings_on_one_item_survive_and_identical_replay_deduplicates(tmp_path, writer):
    from tests.test_review_ledger import _answer

    original = _record(tmp_path)
    findings = [{"item": "rollback", "verdict": "FAIL", "severity": "critical", "reason": reason}
                for reason in ("a.py:1 loses state", "b.py:2 loses a different state")]
    raw = _raw("a", "a", parts=BOTH, operation_id="a",
               answers={"change": _answer("change", "FAIL", findings + findings[:1]), "coupling": _answer("coupling")})
    fresh = ledger.build_commit_gate_record(_facts([raw], critical_findings=ledger.seat_findings([raw], set())[0]),
                                             drive_root=tmp_path).to_dict()
    if writer == "collected":
        merge = lambda: ledger.settle_from_collected(tmp_path, original["record_id"], [raw])
    else:
        merge = lambda: ledger.revise_record(tmp_path, original["record_id"],
                                             lambda now: ledger.settle_pending_payload(now, fresh))
    first, replay = merge(), merge()
    assert first["verdict"]["critical_findings"] == replay["verdict"]["critical_findings"]
    assert {f["reason"] for f in replay["verdict"]["critical_findings"]} == {f["reason"] for f in findings}
    assert len(replay["verdict"]["critical_findings"]) == 2


@pytest.mark.parametrize("operation", ["write", "revise", "collect"])
@pytest.mark.parametrize("lock_failure", ["unavailable", "raises"])
def test_unavailable_lock_writes_nothing_and_normal_lock_recovers(tmp_path, monkeypatch, operation, lock_failure):
    from ouroboros import platform_layer

    original = _record(tmp_path)
    before = {p: p.read_bytes() for p in ledger.ledger_dir(tmp_path).iterdir() if p.is_file()}
    raw = _raw("a", "a", parts=BOTH, operation_id="a")
    if operation == "write":
        mutate = lambda: ledger.write_record(tmp_path, {**original, "revision": 2})
    elif operation == "revise":
        mutate = lambda: ledger.revise_record(tmp_path, original["record_id"], lambda row: {**row, "test_mark": True})
    else:
        mutate = lambda: ledger.settle_from_collected(tmp_path, original["record_id"], [raw])
    def unavailable(*_args, **_kwargs):
        if lock_failure == "raises":
            raise OSError("lock storage unavailable")
        return None
    with monkeypatch.context() as patch:
        patch.setattr(platform_layer, "acquire_exclusive_file_lock", unavailable)
        patch.setattr(ledger, "_write_and_index", lambda *_a: pytest.fail("wrote without acquiring the lock"))
        with pytest.raises(OSError, match="ledger lock unavailable"):
            mutate()
    assert before == {p: p.read_bytes() for p in ledger.ledger_dir(tmp_path).iterdir() if p.is_file()}
    recovered = mutate()
    assert recovered["revision"] == 2
    if operation == "collect":
        assert recovered["rows"][0]["status"] == "responded"
