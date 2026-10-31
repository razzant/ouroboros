"""Pending inventories use typed facts; replay and one-time legacy backfill read exact bodies."""
import pytest

from ouroboros import delegate_custody as custody, delegate_pending, maintenance, store
from ouroboros import delegate_custody_current as current
from ouroboros.observability import write_blob


def request(root, identity):
    return {"type": custody.START_REQUESTED, "task_id": "owner", "invocation_id": identity,
            "request_ref": write_blob(root, {"prompt": "exact assignment", "model": "original",
                                             "access": "readonly", "mode": "ask"}, kind="json")}


def test_start_publication_keeps_request_facts_until_start_or_definite_refusal(tmp_path, monkeypatch):
    for identity in ("started", "refused", "uncertain"):
        event = request(tmp_path, identity)
        assert custody.emit(tmp_path, custody.START_REQUESTED, event)
    monkeypatch.setattr(delegate_pending, "request_body", lambda *a: pytest.fail("inventory read history"))
    assert all(row["request"] is None and row["request_ref"] for row in custody.pending_invocations(tmp_path))
    assert custody.emit(tmp_path, custody.STARTED, {"invocation_id": "started", "run_id": "run", "task_id": "owner"})
    assert custody.emit(tmp_path, custody.START_FAILED, {"invocation_id": "refused", "definite": True})
    assert custody.emit(tmp_path, custody.START_FAILED, {"invocation_id": "uncertain", "definite": False})
    assert [row["invocation_id"] for row in custody.pending_invocations(tmp_path)] == ["uncertain"]


@pytest.mark.parametrize("cached", [False, True])
def test_legacy_backfill_reads_only_open_requests_once_and_preserves_an_unreadable_source(tmp_path, monkeypatch, cached):
    from ouroboros.delegate_registration_policy import independent_review_without_apply

    events = [{**request(tmp_path, identity), "source": "review_substrate"}
              for identity in ("open", "closed", "unknown")]
    events[-1]["request_ref"]["sha256"] = "0" * 64
    events.append({"type": custody.STARTED, "invocation_id": "closed", "run_id": "closed"})
    with store.open_store(tmp_path, write=True) as txn:
        current.write_fold(txn, current.fold(events), replace=True)
        if cached:
            for key, facts in txn.set_rows("custody_open").items():
                txn.put_row("custody_open", key, {**facts, "pending": delegate_pending.pending_invocations(
                    tmp_path, [facts["request"]], materialize=False)})
            txn.set_meta("job:pending_invocations", {"state": "done"})
    original, read = delegate_pending.pending_invocations, []
    def observe(root, rows=None, **kwargs):
        if rows is not None:
            read.extend(row["invocation_id"] for row in rows)
        return original(root, rows, **kwargs)
    monkeypatch.setattr(delegate_pending, "pending_invocations", observe)
    resolve = delegate_pending.request_body
    monkeypatch.setattr(delegate_pending, "request_body", lambda *a: pytest.fail("inventory read a blob"))
    assert {row["invocation_id"] for row in custody.pending_invocations(tmp_path)} == {"open", "unknown"}
    read.clear()  # a legacy reader may project the same typed identity, without a body read
    bodies = []
    def resolve_once(root, row):
        bodies.append(row["invocation_id"])
        return resolve(root, row)
    monkeypatch.setattr(delegate_pending, "request_body", resolve_once)
    assert maintenance.run_job(tmp_path, "pending_invocations")["items"] == 2
    assert sorted(read) == ["open", "unknown"]
    assert sorted(bodies) == ["open", "unknown"]
    monkeypatch.setattr(delegate_pending, "request_body", lambda *a: pytest.fail("completed backfill/inventory read a blob"))
    assert maintenance.run_job(tmp_path, "pending_invocations") is None
    rows = {row["invocation_id"]: row for row in custody.pending_invocations(tmp_path)}
    assert rows["open"]["request"] is None and resolve(tmp_path, rows["open"])["model"] == "original"
    assert rows["unknown"]["request"] is None and rows["unknown"]["request_ref"]
    assert resolve(tmp_path, rows["unknown"]) is None
    assert independent_review_without_apply(rows["open"]) is True
    assert independent_review_without_apply(rows["unknown"]) is False
    assert sorted(read) == ["open", "unknown"]


def test_backfill_cannot_resurrect_a_start_that_wins_during_source_read(tmp_path, monkeypatch):
    with store.open_store(tmp_path, write=True) as txn:
        txn.put_row("custody_open", "invocation:racing", {"request": request(tmp_path, "racing")})
    original = delegate_pending.pending_invocations
    def racing(root, rows=None, **kwargs):
        body = original(root, rows, **kwargs)
        custody.emit(root, custody.STARTED, {"invocation_id": "racing", "run_id": "run"})
        return body
    monkeypatch.setattr(delegate_pending, "pending_invocations", racing)
    maintenance.run_job(tmp_path, "pending_invocations")
    assert custody.pending_invocations(tmp_path) == []


def test_backfill_does_not_mark_a_racing_import_done(tmp_path, monkeypatch):
    with store.open_store(tmp_path, write=True) as txn:
        current.write_fold(txn, current.fold([request(tmp_path, "first")]), replace=True)
    original, raced = delegate_pending.pending_invocations, []
    def arriving(root, rows=None, **kwargs):
        if not raced:
            raced.append(True)
            with store.open_store(root, write=True) as txn:
                current.write_fold(txn, current.fold([request(root, "arriving")]), replace=False)
        return original(root, rows, **kwargs)
    monkeypatch.setattr(delegate_pending, "pending_invocations", arriving)
    assert maintenance.run_job(tmp_path, "pending_invocations")["state"] == "pending"
    assert maintenance.run_job(tmp_path, "pending_invocations")["state"] == "done"
    assert {row["invocation_id"] for row in custody.pending_invocations(tmp_path)} == {"first", "arriving"}


@pytest.mark.parametrize("inline", [None, {}])
def test_an_old_cached_body_never_replaces_the_canonical_reference(tmp_path, monkeypatch, inline):
    from pathlib import Path

    source = {**request(tmp_path, "old-cache"), "request": inline}
    cached = delegate_pending.pending_invocations(tmp_path, [source])
    assert cached[0]["request"]["prompt"] == "exact assignment"
    with store.open_store(tmp_path, write=True) as txn:
        txn.put_row("custody_open", "invocation:old-cache", {"request": source, "pending": cached})
    resolve = delegate_pending.request_body
    monkeypatch.setattr(delegate_pending, "request_body", lambda *a: pytest.fail("inventory resolved history"))
    [pending] = custody.pending_invocations(tmp_path)
    assert pending["request"] is None and pending["request_ref"] == source["request_ref"]
    assert resolve(tmp_path, pending)["prompt"] == "exact assignment"
    Path(source["request_ref"]["path"]).unlink()
    [pending] = custody.pending_invocations(tmp_path)
    assert pending["request"] is None and resolve(tmp_path, pending) is None
