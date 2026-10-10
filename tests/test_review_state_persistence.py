"""Review-state encoding and conditional saves retain the whole locked transaction."""

from __future__ import annotations

import copy
import errno
import json
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import pytest

from ouroboros import review_state as store
from ouroboros.utils import atomic_write_json


@dataclass
class NestedEvidence:
    label: str
    contents: object


def _attempt(**values):
    return store.CommitAttemptRecord(
        ts="2026-10-09T08:00:00+00:00", commit_message="preserve evidence", status="reviewed", **values,
    )


def _path(root):
    return root / "state" / "advisory_review.json"


def _payload(root):
    data = json.loads(_path(root).read_text(encoding="utf-8"))
    data.pop("saved_at")
    return data


def _legacy_payload(state):
    """The prior asdict + indented JSON contract, independent of the new encoder."""
    state = copy.deepcopy(state)
    store._prepare_state_for_persistence(state)
    for attempt in state.attempts:
        if not hasattr(attempt, "author_disposition"):
            attempt.author_disposition = {}
    data = asdict(state)
    data["state_version"] = data["schema_version"] = store._STATE_SCHEMA_VERSION
    data["next_obligation_seq"] = int(state.next_obligation_seq or 1)
    data["next_commit_readiness_debt_seq"] = int(state.next_commit_readiness_debt_seq or 1)
    return json.loads(json.dumps(data, ensure_ascii=False, indent=2))


def _writes(monkeypatch):
    writes = []
    original = store.write_text_atomic

    def write(path, text):
        writes.append(path)
        original(path, text)

    monkeypatch.setattr(store, "write_text_atomic", write)
    return writes


def test_compact_encoding_preserves_full_legacy_payload_and_nested_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_utc_now", lambda: "2026-10-09T09:00:00+00:00")
    raw = {
        "context_manifest": NestedEvidence("полный контекст", {
            "inner": NestedEvidence("nested", (1, None, "é")),
        }),
        "raw_text": "full reviewer answer\nnext line",
        "answers": [{"part": "coupling", "value": "retained"}],
        "source_refs": ["source://exact-subject"],
        "pending_invocation_id": "inv-one",
    }
    attempt = _attempt(triad_raw_results=[raw], scope_raw_result={"raw_results": [copy.deepcopy(raw)]})
    del attempt.author_disposition
    state = store.AdvisoryReviewState(
        state_version=1, attempts=[attempt],
        advisory_runs=[store.AdvisoryRunRecord("snapshot", "message", "stale", "stamp", raw_result="all words")],
        open_obligations=[store.ObligationItem("obl-7", "tests_affected", "critical", "reason", "stamp", "msg")],
        commit_readiness_debts=[store.CommitReadinessDebtItem("crd-9", "review", "keep debt")],
        last_stale_from_edit_ts="stamp", last_stale_reason="changed", last_stale_repo_key="repo",
        last_stale_task_id="editor",
    )
    expected = _legacy_payload(state)

    store.save_state(tmp_path, state)

    assert _payload(tmp_path) == expected
    assert set(_payload(tmp_path)) == {
        "state_version", "schema_version", "advisory_runs", "attempts", "open_obligations",
        "next_obligation_seq", "commit_readiness_debts", "next_commit_readiness_debt_seq",
        "last_stale_from_edit_ts", "last_stale_reason", "last_stale_repo_key", "last_stale_task_id",
    }
    assert attempt.author_disposition == {}
    assert "\n" not in _path(tmp_path).read_text(encoding="utf-8")
    loaded = store._load_state_unlocked(tmp_path, strict_attempt_authority=True)
    assert loaded.attempts[0].triad_raw_results == expected["attempts"][0]["triad_raw_results"]
    assert loaded.next_obligation_seq == 8 and loaded.next_commit_readiness_debt_seq == 10
    # The shared JSON writer keeps its existing formatting contract.
    other = tmp_path / "other.json"
    atomic_write_json(other, {"unchanged": [1, 2]})
    assert "\n" in other.read_text(encoding="utf-8")


def test_unknown_object_is_not_stringified_and_cannot_replace_durable_evidence(tmp_path):
    state = store.AdvisoryReviewState(attempts=[_attempt(triad_raw_results=[{"raw_text": "retained"}])])
    store.save_state(tmp_path, state)
    before = _path(tmp_path).read_bytes()
    state.attempts[0].triad_raw_results[0]["unsupported"] = object()

    with pytest.raises(TypeError, match="not JSON serializable"):
        store.save_state(tmp_path, state)

    assert _path(tmp_path).read_bytes() == before
    state.attempts[0].triad_raw_results[0]["unsupported"] = "supported"
    store.save_state(tmp_path, state)
    assert _payload(tmp_path)["attempts"][0]["triad_raw_results"][0]["unsupported"] == "supported"


@pytest.mark.parametrize("operation", ["save", "update"])
@pytest.mark.parametrize("reason,number", [("contention", None), ("permission", errno.EACCES),
                                           ("kernel_refused", errno.ENOLCK), ("identity_unreadable", None)])
def test_acquisition_refusal_preserves_platform_cause_and_timeout_compatibility(
    tmp_path, monkeypatch, operation, reason, number,
):
    store.save_state(tmp_path, store.AdvisoryReviewState())
    before = _path(tmp_path).read_bytes()
    options = []

    def refuse(path, **kwargs):
        options.append(kwargs)
        kwargs["outcome"].update(reason=reason, errno=number)

    monkeypatch.setattr(store, "acquire_exclusive_file_lock", refuse)
    with pytest.raises(TimeoutError) as caught:
        if operation == "save":
            store.save_state(tmp_path, store.AdvisoryReviewState())
        else:
            store.update_state(tmp_path, lambda _: pytest.fail("mutator ran without its lock"))

    error = caught.value
    assert isinstance(error, store.ReviewStateLockError)
    assert error.lock_outcome["reason"] == reason and error.lock_outcome["errno"] == number
    assert error.lock_outcome["elapsed_sec"] >= 0 and error.lock_outcome["timeout_sec"] == 4.0
    assert json.loads(error.reported_cause) == error.lock_outcome
    assert options[0]["stale_sec"] == 90.0 and "owner_aware_stale" not in options[0]
    assert _path(tmp_path).read_bytes() == before


@pytest.mark.parametrize("value", [None, False, 0, [], "read result"])
def test_only_explicit_unchanged_results_skip_a_save_and_keep_return_contract(tmp_path, monkeypatch, value):
    store.save_state(tmp_path, store.AdvisoryReviewState())
    before, mtime = _path(tmp_path).read_bytes(), _path(tmp_path).stat().st_mtime_ns
    writes = _writes(monkeypatch)

    unchanged = store.update_state(tmp_path, lambda _: store.ReviewStateMutation(value, changed=False))

    assert isinstance(unchanged, store.AdvisoryReviewState) if value is None else unchanged == value
    assert writes == [] and _path(tmp_path).read_bytes() == before
    assert _path(tmp_path).stat().st_mtime_ns == mtime
    ordinary = store.update_state(tmp_path, lambda _: value)
    assert isinstance(ordinary, store.AdvisoryReviewState) if value is None else ordinary == value
    assert writes == [_path(tmp_path)]  # arbitrary false/None still keeps the old save behavior


@pytest.mark.parametrize("legacy", ["missing_field", "coalescing", "hydration", "roster", "ordering", "counter_type"])
def test_false_changed_fact_still_persists_load_and_preparation_changes(tmp_path, monkeypatch, legacy):
    state = store.AdvisoryReviewState(attempts=[_attempt()])
    store.save_state(tmp_path, state)
    raw = json.loads(_path(tmp_path).read_text(encoding="utf-8"))
    if legacy == "missing_field":
        del raw["attempts"][0]["author_disposition"]
    elif legacy == "coalescing":
        obligation = asdict(store.ObligationItem("obl-7", "tests_affected", "critical", "same", "old", "m"))
        raw["open_obligations"] = [obligation, {**obligation, "obligation_id": "obl-8"}]
    elif legacy == "hydration":
        raw["commit_readiness_debts"] = [asdict(store.CommitReadinessDebtItem("crd-9", "review", "debt"))]
    elif legacy == "roster":
        raw["attempts"][0]["triad_raw_results"] = "malformed legacy roster"
    elif legacy == "ordering":
        older = {**raw["attempts"][0], "ts": "2000-01-01T00:00:00+00:00"}
        raw["attempts"].append(older)
    else:
        raw["next_obligation_seq"] = True
    _path(tmp_path).write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(store, "_utc_now", lambda: "2026-10-09T09:00:00+00:00")
    expected = _legacy_payload(store._load_state_unlocked(tmp_path, strict_attempt_authority=True))
    writes = _writes(monkeypatch)

    store.update_state(tmp_path, lambda _: store.ReviewStateMutation(False, changed=False))

    assert _payload(tmp_path) == expected and writes == [_path(tmp_path)]
    assert type(_payload(tmp_path)["next_obligation_seq"]) is int
    store.update_state(tmp_path, lambda _: store.ReviewStateMutation(False, changed=False))
    assert writes == [_path(tmp_path)]  # once normalized, the identical transaction skips


def test_original_payload_does_not_alias_nested_rows_changed_in_place(tmp_path, monkeypatch):
    store.save_state(tmp_path, store.AdvisoryReviewState(attempts=[_attempt(
        triad_raw_results=[{"context_manifest": {"answers": ["first"]}}],
    )]))
    writes = _writes(monkeypatch)

    def mutate(state):
        state.attempts[0].triad_raw_results[0]["context_manifest"]["answers"].append("second")
        return store.ReviewStateMutation(None, changed=False)

    result = store.update_state(tmp_path, mutate)

    assert isinstance(result, store.AdvisoryReviewState)
    assert _payload(tmp_path)["attempts"][0]["triad_raw_results"][0]["context_manifest"]["answers"] == ["first", "second"]
    assert writes == [_path(tmp_path)]


def test_conditional_save_cannot_bypass_strict_authority_load(tmp_path):
    _path(tmp_path).parent.mkdir()
    _path(tmp_path).write_text('{"attempts":[{"paid":"yes"}]}', encoding="utf-8")
    before = _path(tmp_path).read_bytes()
    with pytest.raises(ValueError, match="invalid paid"):
        store.update_state(tmp_path, lambda _: pytest.fail("strict load must precede the no-op mutator"))
    assert _path(tmp_path).read_bytes() == before


def test_marker_noop_keeps_first_editor_but_other_checkout_and_new_look_still_write(tmp_path, monkeypatch):
    from ouroboros import review_ledger

    repo_a, repo_b = tmp_path / "a", tmp_path / "b"
    repo_a.mkdir()
    repo_b.mkdir()
    drive = tmp_path / "drive"
    key_a, key_b = store.make_repo_key(repo_a), store.make_repo_key(repo_b)
    state = store.AdvisoryReviewState(
        last_stale_from_edit_ts="2026-10-09T09:00:00+00:00", last_stale_repo_key=key_a,
        last_stale_reason="first mutation", last_stale_task_id="first-editor",
    )
    store.save_state(drive, state)
    look = {"ts": "2026-10-09T08:00:00+00:00"}
    monkeypatch.setattr(review_ledger, "latest_preflight_record", lambda *_a, **_kw: look)
    monkeypatch.setattr(store, "_utc_now", lambda: "2026-10-09T10:00:00+00:00")
    writes = _writes(monkeypatch)

    store.invalidate_advisory_after_mutation(drive, mutation_root=repo_a, mutating_task_id="later-editor")
    assert writes == [] and store.load_state(drive).last_stale_task_id == "first-editor"

    acquire = store.acquire_review_state_lock
    intervene = [True]

    def other_checkout_before_acquisition(*args, **kwargs):
        if intervene:
            intervene.pop()
            other = store.load_state(drive)
            other.last_stale_repo_key, other.last_stale_task_id = key_b, "other-checkout-editor"
            store.save_state(drive, other)
        return acquire(*args, **kwargs)

    monkeypatch.setattr(store, "acquire_review_state_lock", other_checkout_before_acquisition)
    store.invalidate_advisory_after_mutation(drive, mutation_root=repo_a, mutating_task_id="current-editor")
    current = store.load_state(drive)
    assert (current.last_stale_repo_key, current.last_stale_task_id) == (key_a, "current-editor")
    assert len(writes) == 2  # the intervening writer and A's current locked predicate

    look["ts"] = "2026-10-09T10:30:00+00:00"
    monkeypatch.setattr(store, "_utc_now", lambda: "2026-10-09T11:00:00+00:00")
    store.invalidate_advisory_after_mutation(drive, mutation_root=repo_a, mutating_task_id="new-look-editor")
    assert len(writes) == 3 and store.load_state(drive).last_stale_task_id == "new-look-editor"


@pytest.mark.parametrize("tool", ["review_change", "commit_reviewed"])
def test_expiration_callers_skip_reads_but_persist_real_expiration(tmp_path, monkeypatch, tool):
    from ouroboros.tools.commit_gate import _check_overlapping_review_attempt
    from ouroboros.tools.review_change_custody import pending_round_attempt

    repo = tmp_path / "repo"
    repo.mkdir()
    ctx = SimpleNamespace(drive_root=tmp_path / "drive", repo_dir=repo, task_id="task")
    attempt = _attempt(repo_key=store.make_repo_key(repo), tool_name=tool, task_id="task", review_retry_key="round")
    attempt.status = "reviewing"
    store.save_state(ctx.drive_root, store.AdvisoryReviewState(attempts=[attempt]))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    monkeypatch.setattr(store, "_utc_now", lambda: "2026-10-09T08:01:00+00:00")
    writes = _writes(monkeypatch)

    def read():
        return (pending_round_attempt(ctx, root=repo, retry_key="round") if tool == "review_change"
                else _check_overlapping_review_attempt(ctx))

    assert read() is not None and writes == []
    monkeypatch.setattr(store, "_utc_now", lambda: "2026-10-09T10:00:00+00:00")
    assert read() is None and writes == [_path(ctx.drive_root)]
    expired = store.load_state(ctx.drive_root).attempts[0]
    assert expired.status == "failed" and expired.phase == "expired"
    assert read() is None and writes == [_path(ctx.drive_root)]
