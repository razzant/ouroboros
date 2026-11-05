"""A review-state save keeps a roster inventory as a ref to its producer payload."""

from __future__ import annotations

import copy
import hashlib
import json

from ouroboros import review_state as store


def _attempt(**values):
    return store.CommitAttemptRecord(
        ts="2026-10-09T08:00:00+00:00", commit_message="preserve evidence", status="reviewed", **values,
    )


def _inventory():
    return [
        {"path": "a.py", "size": 3, "sha256": "ab", "mode": "100644"},
        {"path": "б.py", "size": 1, "sha256": "cd", "mode": "100644"},
    ]


def _evidence(named):
    return {
        "status": "complete", "evidence_dir": "/tmp/ev", "source": "seat",
        "gaps": ["one"], "patch_facts": {"files": 2}, "named_evidence": named,
    }


def _row(named, response_ref=None, **extra):
    row = {
        "slot_id": "triad_0", "status": "pass", "raw_text": "kept",
        "response_ref": {"call_id": "call-1", "manifest_ref": "manifest-1"} if response_ref is None else response_ref,
        "usage": {"prompt_tokens": 3, "checking_evidence": _evidence(named)},
        "parsed": {"keep": True},
    }
    row.update(extra)
    return row


def _expected_ref(named, response_ref):
    canonical = json.dumps(named, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return {
        "retained_in": response_ref,
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "bytes": len(canonical),
        "file_count": len(named) if isinstance(named, list) else 0,
    }


def test_projection_is_idempotent_and_does_not_mutate_the_row():
    inventory = _inventory()
    response_ref = {"call_id": "call-1"}
    row = _row(inventory, response_ref=response_ref)
    original = copy.deepcopy(row)
    once = store._project_named_evidence(row)
    assert store._project_named_evidence(once) is once
    assert row == original
    assert row["usage"]["checking_evidence"]["named_evidence"] is inventory
    evidence = once["usage"]["checking_evidence"]
    assert evidence["named_evidence"] == _expected_ref(inventory, response_ref)
    assert evidence["named_evidence"]["retained_in"] is response_ref
    assert evidence["status"] == "complete" and evidence["gaps"] == ["one"]
    assert evidence["evidence_dir"] == "/tmp/ev" and evidence["patch_facts"] == {"files": 2}
    assert once["parsed"] == {"keep": True} and once["usage"]["prompt_tokens"] == 3
    assert once["raw_text"] == "kept" and once["response_ref"] is response_ref


def test_rows_without_a_response_ref_stay_unchanged():
    inventory = _inventory()
    missing = _row(inventory)
    del missing["response_ref"]
    blank = _row(inventory, response_ref="  ")
    empty_ref = _row(inventory, response_ref="")
    empty_inventory = _row([])
    absent_inventory = _row(None)
    malformed_scope = {
        "raw_results": "bad", "response_ref": {"call_id": "x"},
        "usage": {"checking_evidence": {"named_evidence": inventory}},
    }
    for row in (missing, blank, empty_ref, empty_inventory, absent_inventory):
        assert store._project_named_evidence(row) is row
    assert store._project_scope(malformed_scope) is malformed_scope
    assert missing["usage"]["checking_evidence"]["named_evidence"] == inventory
    assert empty_inventory["usage"]["checking_evidence"]["named_evidence"] == []


def test_load_save_load_keeps_every_other_field(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_utc_now", lambda: "2026-10-09T09:00:00+00:00")
    inventory = _inventory()
    response_ref = {"call_id": "call-1", "manifest_ref": "manifest-1"}
    kept = _row(copy.deepcopy(inventory))
    del kept["response_ref"]
    projected = _row(inventory, response_ref=response_ref)
    scope_row = _row(copy.deepcopy(inventory), response_ref={"call_id": "scope"})
    scope_plain = {"status": "pass", "raw_text": "scope kept"}
    historical = _row(copy.deepcopy(inventory), response_ref={"call_id": "hist"})
    attempt = _attempt(
        triad_raw_results=[projected, kept, "not-a-row"],
        scope_raw_result={"raw_results": [scope_row, scope_plain], "context_manifest": {"keep": 1}},
    )
    historical_attempt = _attempt(scope_raw_result=historical)
    store.save_state(tmp_path, store.AdvisoryReviewState(attempts=[attempt, historical_attempt]))
    assert projected["usage"]["checking_evidence"]["named_evidence"] is inventory

    loaded = store.load_state(tmp_path)
    saved = loaded.attempts[0].triad_raw_results[0]
    assert saved["usage"]["checking_evidence"]["named_evidence"] == _expected_ref(inventory, response_ref)
    restored = copy.deepcopy(saved)
    restored["usage"]["checking_evidence"]["named_evidence"] = inventory
    assert restored == projected
    assert loaded.attempts[0].triad_raw_results[1] == kept
    assert loaded.attempts[0].triad_raw_results[2] == "not-a-row"
    scope = loaded.attempts[0].scope_raw_result
    assert scope["context_manifest"] == {"keep": 1}
    assert scope["raw_results"][1] == scope_plain
    assert scope["raw_results"][0]["usage"]["checking_evidence"]["status"] == "complete"
    assert scope["raw_results"][0]["usage"]["checking_evidence"]["named_evidence"]["file_count"] == 2
    historical_saved = loaded.attempts[1].scope_raw_result
    assert "raw_results" not in historical_saved
    assert historical_saved["usage"]["checking_evidence"]["named_evidence"]["retained_in"] == {"call_id": "hist"}
    assert historical_saved["raw_text"] == "kept" and historical_saved["parsed"] == {"keep": True}


def test_noop_save_writes_nothing_after_the_first_save(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_utc_now", lambda: "2026-10-09T09:00:00+00:00")
    store.save_state(tmp_path, store.AdvisoryReviewState(attempts=[_attempt(triad_raw_results=[_row(_inventory())])]))
    path = tmp_path / "state" / "advisory_review.json"
    before = path.read_bytes()
    writes = []
    original = store.write_text_atomic

    def write(target, text):
        writes.append(target)
        original(target, text)

    monkeypatch.setattr(store, "write_text_atomic", write)
    store.update_state(tmp_path, lambda _state: store.ReviewStateMutation(None, changed=False))
    assert writes == []
    assert path.read_bytes() == before
