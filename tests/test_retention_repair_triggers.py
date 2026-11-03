"""Immutable input custody and cheap retries of unchanged unavailable sources."""
import gzip
import json
from pathlib import Path

import pytest

from ouroboros import artifacts, headless, maintenance_jobs, observability, source_retention, task_custody
from ouroboros.task_results import load_task_result, write_task_result
from tests._store_shared import result_signature


@pytest.mark.parametrize("count", [1, 26])
def test_persisted_input_contract_retains_exact_sources_after_gc(tmp_path, count):
    parent, task = tmp_path / "canonical", "input-contract"
    child = headless.prepare_task_drive(parent, task, "empty")
    inputs = []
    for index in range(count):
        path = tmp_path / f"input-{index}.txt"
        path.write_text(f"exact input {index}", encoding="utf-8")
        inputs.append(str(path))
    manifest = artifacts.stage_task_attachments(child, task, inputs)
    contract = artifacts.attachment_manifest_projection(child, task, manifest)
    write_task_result(child, task, "running", result="answer", task_contract=contract)
    write_task_result(parent, task, "running", child_drive_root=str(child))

    # An older persisted acceptance request (the retired task acceptance wrote these) names the
    # same contract; an old record still promotes whole, its captured bytes retained as written.
    raw = json.dumps({"request": {"surface": "task_acceptance", "task_id": task,
                                  "evidence": {"task_contract": contract}}}).encode("utf-8")
    captured = artifacts.store_actor_source_bytes(child, task, category="context_checkpoints",
        source_id="acceptance", data=raw, extension="json")
    write_task_result(child, task, "completed", result="answer", artifact_status="ready", task_contract=contract,
                      review_evidence={"source_refs": [captured]})
    headless.copy_child_task_result(parent, {"id": task, "drive_root": str(child)})
    result = headless.retry_child_task_refs(parent, child, task)
    assert result["child_ref_promotion"]["status"] == "complete"
    assert not result["child_ref_promotion"]["unavailable_refs"]
    assert artifacts.read_actor_source_bytes(parent, task, captured) == raw
    assert task_custody.settle_child_drive(parent, task, child, live=lambda _: False)["status"] == "removed"
    assert artifacts.read_actor_source_bytes(parent, task, captured) == raw
    rows = artifacts.resolve_attachment_manifest(parent, task, result["task_contract"])
    assert len(rows) == count
    for row in rows:
        artifacts.stream_artifact_file(Path(row["abs_path"]), expected=row)
        assert Path(row["abs_path"]).read_text(encoding="utf-8").startswith("exact input ")


def _inventory(tmp_path, fault="missing"):
    parent, task = tmp_path / "canonical", "inventory-repair"
    child = headless.prepare_task_drive(parent, task, "empty")
    for index in range(3):
        observability.persist_call(child, task_id=task, call_id=f"healthy-{index}", call_type="tool_call",
                                   payload={"tool": "read_file", "result": f"exact {index}"})
    call = observability.persist_call(child, task_id=task, call_id="broken", call_type="tool_call",
                                      payload={"tool": "read_file", "result": "repairable original"})
    manifest = json.loads(Path(call["manifest_ref"]["path"]).read_text(encoding="utf-8"))
    blob = Path(manifest["full_payload_ref"]["path"])
    exact = blob.read_bytes()
    if fault == "missing":
        blob.unlink()
    elif fault == "corrupt":
        blob.write_bytes(gzip.compress(b'{"different": "captured bytes"}'))
    elif fault == "gzip_corrupt":
        blob.write_bytes(b"not a gzip stream")
    elif fault == "gzip_deflate":
        blob.write_bytes(_invalid_deflate(exact))
    write_task_result(child, task, "completed", result="answer", artifact_status="ready")
    headless.copy_child_task_result(parent, {"id": task, "drive_root": str(child)})
    return parent, child, task, blob, exact


def _observe(monkeypatch):
    from ouroboros import utils

    counts = {"reads": 0, "writes": 0, "events": 0}
    read, write, append = source_retention.RetentionWalk._read_source, observability.write_call_manifest, utils.append_jsonl
    def read_source(self, node):
        counts["reads"] += 1
        return read(self, node)
    def write_manifest(*args, **kwargs):
        counts["writes"] += 1
        return write(*args, **kwargs)
    def append_event(path, row, **kwargs):
        if row.get("type") == "history_retention":
            counts["events"] += 1
        return append(path, row, **kwargs)
    monkeypatch.setattr(source_retention.RetentionWalk, "_read_source", read_source)
    monkeypatch.setattr(observability, "write_call_manifest", write_manifest)
    monkeypatch.setattr(utils, "append_jsonl", append_event)
    return counts


@pytest.mark.parametrize("fault", ["missing", "corrupt", "gzip_corrupt", "gzip_deflate"])
def test_unchanged_missing_inventory_waits_for_real_repair_without_rewalk_or_gc(tmp_path, monkeypatch, fault):
    parent, child, task, blob, exact = _inventory(tmp_path, fault)
    counts = _observe(monkeypatch)
    assert observability.retry_pending_child_ref_promotions(parent)["pending"] == [task]
    state = load_task_result(parent, task)["child_ref_promotion"]
    assert state["call_inventory_preserved"] is False
    assert state["pending_refs"][-1]["reason"] == "call_inventory_unavailable"
    assert state["unavailable_refs"][0]["reason"] == {
        "missing": "source_missing", "corrupt": "digest_mismatch",
        "gzip_corrupt": "source_unreadable", "gzip_deflate": "source_unreadable"}[fault]
    if fault == "gzip_deflate":
        assert state["unavailable_refs"][0]["source_error_type"] == "zlib.error"
    assert task_custody.settle_child_drive(parent, task, child, live=lambda _: False)["status"] == "retained"
    first = dict(counts)
    calls = child / "observability/calls" / task
    revision = calls.stat().st_mtime_ns
    projections = {path: (path.stat().st_ino, path.stat().st_mtime_ns)
                   for path in (parent / "observability/calls" / task).glob("*.json")}
    report = observability.retry_pending_child_ref_promotions(parent)
    assert report["unchanged"] == [task] and not report["retried"]
    assert counts == first
    assert all((path.stat().st_ino, path.stat().st_mtime_ns) == fact for path, fact in projections.items())
    blob.write_bytes(exact)  # Repair a leaf without changing the call inventory.
    assert calls.stat().st_mtime_ns == revision
    assert observability.retry_pending_child_ref_promotions(parent)["completed"] == [task]
    assert counts["reads"] > first["reads"] and counts["events"] == first["events"] + 1
    assert all((path.stat().st_ino, path.stat().st_mtime_ns) == fact for path, fact in projections.items())
    assert task_custody.settle_child_drive(parent, task, child, live=lambda _: False)["status"] == "removed"
    observability.retry_pending_child_ref_promotions(parent)
    assert task not in (maintenance_jobs.job(parent, source_retention.MEMO_JOB) or {}).get("observed", {})


def test_explicit_retry_metadata_new_call_and_new_release_recheck_sources(tmp_path, monkeypatch):
    parent, child, task, _blob, _exact = _inventory(tmp_path)
    counts = _observe(monkeypatch)
    observability.retry_pending_child_ref_promotions(parent)
    previous = counts["reads"]
    headless.retry_child_task_refs(parent, child, task)  # Direct calls never use the negative observation.
    assert counts["reads"] > previous
    for change in ("metadata", "call", "release"):
        previous = counts["reads"]
        if change == "metadata":
            write_task_result(parent, task, "completed", metadata={"new": "owner fact"})
        elif change == "call":
            observability.persist_call(child, task_id=task, call_id="new-call", call_type="tool_call", payload={"result": "new"})
        else:
            monkeypatch.setattr(source_retention, "memo_epoch", lambda: "a newer release")
        assert observability.retry_pending_child_ref_promotions(parent)["retried"] == [task]
        assert counts["reads"] > previous
        unchanged = counts["reads"]
        assert observability.retry_pending_child_ref_promotions(parent)["unchanged"] == [task]
        assert counts["reads"] == unchanged


def test_destination_write_failure_retries_but_identical_problem_event_does_not_repeat(tmp_path, monkeypatch):
    parent, child, task, _blob, _exact = _inventory(tmp_path, "healthy")
    counts = _observe(monkeypatch)
    with monkeypatch.context() as fail:
        fail.setattr(observability, "write_call_manifest", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk refused write")))
        observability.retry_pending_child_ref_promotions(parent)
        first = dict(counts)
        assert observability.retry_pending_child_ref_promotions(parent)["retried"] == [task]
        assert counts["reads"] > first["reads"] and counts["events"] == first["events"]
    assert observability.retry_pending_child_ref_promotions(parent)["completed"] == [task]


def test_transient_source_io_retries_without_a_file_metadata_change(tmp_path, monkeypatch):
    parent, _child, task, _blob, _exact = _inventory(tmp_path, "healthy")
    attempts = []
    original = source_retention.RetentionWalk._read_source
    def unavailable(self, node):
        if node["kind"] == "blob":
            attempts.append(node["ref"]["path"])
            raise OSError("temporary source read I/O failure")
        return original(self, node)
    with monkeypatch.context() as fail:
        fail.setattr(source_retention.RetentionWalk, "_read_source", unavailable)
        observability.retry_pending_child_ref_promotions(parent)
        first = len(attempts)
        assert observability.retry_pending_child_ref_promotions(parent)["retried"] == [task]
        assert len(attempts) > first
    assert observability.retry_pending_child_ref_promotions(parent)["completed"] == [task]


def test_closed_generation_retries_without_remembering_partial_work(tmp_path, monkeypatch):
    parent, child, task, _blob, _exact = _inventory(tmp_path, "healthy")
    counts = _observe(monkeypatch)
    report = observability.retry_pending_child_ref_promotions(parent, stop=lambda: counts["reads"] >= 1)
    assert report["deferred"] == [task] and child.exists()
    assert observability.retry_pending_child_ref_promotions(parent)["completed"] == [task]


def test_failed_retention_log_append_is_not_cached_as_delivered(tmp_path, monkeypatch):
    from ouroboros import utils

    parent, _child, task, _blob, _exact = _inventory(tmp_path)
    attempts, append = [], utils.append_jsonl
    def fail_once(path, row, **kwargs):
        if row.get("type") == "history_retention":
            attempts.append(row)
            if len(attempts) == 1:
                return False
        return append(path, row, **kwargs)
    monkeypatch.setattr(utils, "append_jsonl", fail_once)
    observability.retry_pending_child_ref_promotions(parent)
    assert observability.retry_pending_child_ref_promotions(parent)["unchanged"] == [task]
    observability.retry_pending_child_ref_promotions(parent)
    assert len(attempts) == 2


def test_outside_locator_stays_disclosed_without_reads_until_exact_local_version_exists(tmp_path, monkeypatch):
    parent, child, task, _blob, _exact = _inventory(tmp_path, "healthy")
    outside = observability.write_blob(tmp_path / "foreign", {"data": "outside source"})
    forbidden = Path(outside["path"])
    exact = forbidden.read_bytes()
    call = child / "observability/calls" / task / "broken.json"
    manifest = json.loads(call.read_text(encoding="utf-8"))
    manifest.update(full_payload_ref=outside, redacted_projection_ref=outside)
    call.write_text(json.dumps(manifest), encoding="utf-8")
    original_open = Path.open
    def scoped_open(path, *args, **kwargs):
        if path == forbidden:
            pytest.fail("retention read an arbitrary outside source")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", scoped_open)
    counts = _observe(monkeypatch)
    observability.retry_pending_child_ref_promotions(parent)
    state = load_task_result(parent, task)["child_ref_promotion"]
    assert state["unavailable_refs"][0]["reason"] == "invalid_scope"
    first = dict(counts)
    assert observability.retry_pending_child_ref_promotions(parent)["unchanged"] == [task]
    assert counts == first
    local = parent / "observability/blobs" / forbidden.name
    local.write_bytes(exact)  # A locally retained exact version grants no outside read.
    assert observability.retry_pending_child_ref_promotions(parent)["completed"] == [task]
    assert task_custody.settle_child_drive(parent, task, child, live=lambda _: False)["status"] == "removed"


def test_stop_during_basis_check_prevents_retry_of_failed_diagnostic_append(tmp_path, monkeypatch):
    from ouroboros import utils

    parent, _child, task, _blob, _exact = _inventory(tmp_path)
    attempts, append = [], utils.append_jsonl
    def fail_first(path, row, **kwargs):
        if row.get("type") == "history_retention":
            attempts.append(row)
            if len(attempts) == 1:
                return False
        return append(path, row, **kwargs)
    monkeypatch.setattr(utils, "append_jsonl", fail_first)
    observability.retry_pending_child_ref_promotions(parent)
    assert len(attempts) == 1
    closed, path_fact = [], source_retention._path_fact
    def close_during_basis(path):
        fact = path_fact(path)
        closed.append(True)
        return fact
    with monkeypatch.context() as closing:
        closing.setattr(source_retention, "_path_fact", close_during_basis)
        report = observability.retry_pending_child_ref_promotions(parent, stop=lambda: bool(closed))
    assert report["deferred"] == [task] and closed
    assert len(attempts) == 1, "a failed prior append is no permission to write after the run closed"
    observability.retry_pending_child_ref_promotions(parent)
    assert len(attempts) == 2  # An open run can still publish the actual problem.


def test_canonical_transient_io_is_not_reclassified_as_missing_child_or_negative_cached(tmp_path, monkeypatch):
    import errno

    parent, child, task, blob, exact = _inventory(tmp_path)
    local = parent / "observability/blobs" / blob.name
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(exact)
    identity = source_retention._path_fact(local)
    opened, gzip_open = [], gzip.open
    def transient_open(path, mode="rb", *args, **kwargs):
        if Path(path) == local and "r" in mode:
            opened.append(path)
            raise OSError(errno.EIO, "temporary canonical read failure")
        return gzip_open(path, mode, *args, **kwargs)
    with monkeypatch.context() as failing:
        failing.setattr(gzip, "open", transient_open)
        observability.retry_pending_child_ref_promotions(parent)
        state = load_task_result(parent, task)["child_ref_promotion"]
        assert state["unavailable_refs"][0]["reason"] == "source_unreadable"
        assert state["unavailable_refs"][0]["source_error_type"] == "OSError"
        first = len(opened)
        assert observability.retry_pending_child_ref_promotions(parent)["retried"] == [task]
        assert len(opened) > first
    assert not blob.exists() and source_retention._path_fact(local) == identity
    assert observability.retry_pending_child_ref_promotions(parent)["completed"] == [task]
    assert task_custody.settle_child_drive(parent, task, child, live=lambda _: False)["status"] == "removed"


def _invalid_deflate(exact):
    # Generated gzip headers may carry the original filename. Keep the header
    # and trailer, but use a reserved deflate block type in the body.
    header_end = exact.index(b"\x00", 10) + 1 if exact[3] & 8 else 10
    return exact[:header_end] + b"\x07\xff\xff\xff" + exact[-8:]


def test_corrupt_canonical_deflate_recovers_from_healthy_retained_child(tmp_path):
    parent, child, task, blob, exact = _inventory(tmp_path, "healthy")
    target = parent / "observability/blobs" / blob.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_invalid_deflate(exact))
    report = observability.retry_pending_child_ref_promotions(parent)
    assert report["completed"] == [task] and not report["errors"]
    assert gzip.decompress(target.read_bytes()) == gzip.decompress(exact)
    assert task_custody.settle_child_drive(parent, task, child, live=lambda _: False)["status"] == "removed"
    assert gzip.decompress(target.read_bytes()) == gzip.decompress(exact)


def _immutable_artifact(tmp_path):
    parent, task = tmp_path / "canonical", "artifact-repair"
    child = headless.prepare_task_drive(parent, task, "empty")
    source = artifacts.task_artifact_dir_path(child, task, create=True) / "captured.txt"
    exact = b"the immutable captured bytes"
    source.write_bytes(exact)
    artifact = {**artifacts.artifact_record(source), "immutable": True}
    source.write_bytes(b"x" * len(exact))
    write_task_result(child, task, "completed", result="answer", artifact_status="ready", artifacts=[artifact])
    headless.copy_child_task_result(parent, {"id": task, "drive_root": str(child)})
    return parent, child, task, source, exact


@pytest.mark.parametrize("repair_at", ["source", "canonical"])
def test_immutable_artifact_failure_reuses_observation_and_reopens_on_exact_repair(tmp_path, monkeypatch, repair_at):
    parent, child, task, source, exact = _immutable_artifact(tmp_path)
    copies, original = [], artifacts.copy_artifact_file
    monkeypatch.setattr(artifacts, "copy_artifact_file", lambda *a, **kw: (copies.append(a), original(*a, **kw))[1])
    assert observability.retry_pending_child_ref_promotions(parent)["pending"] == [task]
    before, first = result_signature(parent, task), len(copies)
    failure = load_task_result(parent, task)["child_ref_promotion"]["pending_refs"][0]
    assert failure["failure_kind"] == "immutable_identity_mismatch"
    assert task_custody.settle_child_drive(parent, task, child, live=lambda _: False)["status"] == "retained"
    assert observability.retry_pending_child_ref_promotions(parent)["unchanged"] == [task]
    assert len(copies) == first and result_signature(parent, task) == before and child.exists()
    target = source if repair_at == "source" else Path(failure["canonical_path"])
    target.write_bytes(exact)
    assert observability.retry_pending_child_ref_promotions(parent)["completed"] == [task]
    assert len(copies) > first
    row = load_task_result(parent, task)
    assert Path(row["artifacts"][0]["path"]).read_bytes() == exact
    assert not row["child_ref_promotion"]["pending_refs"]
    if repair_at == "source":
        assert task_custody.settle_child_drive(parent, task, child, live=lambda _: False)["status"] == "removed"


def test_immutable_artifact_retry_keeps_release_and_reference_repair_inputs(tmp_path, monkeypatch):
    parent, child, task, _source, _exact = _immutable_artifact(tmp_path)
    observability.retry_pending_child_ref_promotions(parent)
    for change in ("ref", "inventory", "release"):
        if change == "ref":
            row = load_task_result(parent, task)
            row["artifacts"][0]["sha256"] = "f" * 64
            write_task_result(parent, task, "completed", artifacts=row["artifacts"])
        elif change == "inventory":
            observability.persist_call(child, task_id=task, call_id="repair-note", call_type="tool_call", payload={"result": "known"})
        else:
            monkeypatch.setattr(source_retention, "memo_epoch", lambda: "a newer release")
        assert observability.retry_pending_child_ref_promotions(parent)["retried"] == [task]
        assert observability.retry_pending_child_ref_promotions(parent)["unchanged"] == [task]
    assert child.exists() and load_task_result(parent, task)["child_ref_promotion"]["pending_refs"]


def test_transient_artifact_copy_error_is_never_a_stable_failure(tmp_path, monkeypatch):
    parent, _child, task, source, exact = _immutable_artifact(tmp_path)
    source.write_bytes(exact)
    attempts = []
    def unavailable(*args, **kwargs):
        attempts.append(args)
        raise OSError("temporary destination write failure")
    with monkeypatch.context() as failure:
        failure.setattr(artifacts, "copy_artifact_file", unavailable)
        observability.retry_pending_child_ref_promotions(parent)
        first = len(attempts)
        assert observability.retry_pending_child_ref_promotions(parent)["retried"] == [task]
        assert len(attempts) > first
    assert observability.retry_pending_child_ref_promotions(parent)["completed"] == [task]


def test_repair_between_failed_artifact_read_and_memo_cannot_be_cached(tmp_path, monkeypatch):
    parent, _child, task, source, exact = _immutable_artifact(tmp_path)
    remember = source_retention.remember_unavailable_retry
    def repair_then_remember(*args, **kwargs):
        source.write_bytes(exact)
        return remember(*args, **kwargs)
    with monkeypatch.context() as repairing:
        repairing.setattr(source_retention, "remember_unavailable_retry", repair_then_remember)
        assert observability.retry_pending_child_ref_promotions(parent)["pending"] == [task]
    assert observability.retry_pending_child_ref_promotions(parent)["completed"] == [task]


def test_in_place_call_inventory_repair_reopens_unavailable_observation(tmp_path):
    parent, child, task, _blob, _exact = _inventory(tmp_path)
    observability.retry_pending_child_ref_promotions(parent)
    assert observability.retry_pending_child_ref_promotions(parent)["unchanged"] == [task]
    call = child / "observability/calls" / task / "broken.json"
    before = call.parent.stat().st_mtime_ns
    manifest = json.loads(call.read_text(encoding="utf-8"))
    repaired = observability.write_blob(child, {"result": "repaired call source"})
    manifest.update(full_payload_ref=repaired, redacted_projection_ref=repaired)
    call.write_text(json.dumps(manifest), encoding="utf-8")
    assert call.parent.stat().st_mtime_ns == before
    assert observability.retry_pending_child_ref_promotions(parent)["completed"] == [task]


# ---------------------------------------------------------------------------
# Owner Q2=A (14 Oct): what was found unresolvable is not rechecked until its source changes.
# The 14 Oct stall: five tasks re-promoted every 300 s (10-20 min passes) because a review
# attachment placed under ``review_inputs/request-*`` was refused and the refusal was never
# remembered. The placement is owned now; any other deterministic contract refusal is
# remembered by basis, while transient I/O and contention stay retryable.
# ---------------------------------------------------------------------------

def _acceptance_checkpoint(child, task, contract):
    raw = json.dumps({"request": {"surface": "task_acceptance", "task_id": task,
                                  "evidence": {"task_contract": contract}}}).encode("utf-8")
    return artifacts.store_actor_source_bytes(child, task, category="context_checkpoints",
                                              source_id="acceptance", data=raw, extension="json")


def test_review_input_attachment_copies_are_owned_sources(tmp_path):
    """An older persisted request names a reviewer's attachment copy under
    ``source_handles/review_inputs/request-*`` of the task's canonical store, where the retired
    acceptance's request-source closure placed it; nothing places one now, and the old record
    still promotes completely."""
    parent, task = tmp_path / "canonical", "att-snapshot-task"
    child = headless.prepare_task_drive(parent, task, "empty")
    image = tmp_path / "screen.png"
    image.write_bytes(b"fake-png-bytes")
    contract = artifacts.attachment_manifest_projection(
        child, task, artifacts.stage_task_attachments(child, task, [str(image)]))
    write_task_result(child, task, "running", result="answer", task_contract=contract)
    write_task_result(parent, task, "running", child_drive_root=str(child))
    [row] = contract["attachment_manifest"]
    staged = Path(row["abs_path"])
    copy = (artifacts.task_artifact_dir_path(parent, task, create=True) / "source_handles" / "review_inputs"
            / "request-old" / "task_results" / "artifacts" / task / "attachments" / staged.name)
    copy.parent.mkdir(parents=True)
    copy.write_bytes(staged.read_bytes())
    placed = {**contract, "attachment_manifest": [{**row, "abs_path": str(copy)}]}
    captured = _acceptance_checkpoint(child, task, placed)  # the persisted request names the placement
    write_task_result(child, task, "completed", result="answer", artifact_status="ready",
                      task_contract=contract, review_evidence={"source_refs": [captured]})
    headless.copy_child_task_result(parent, {"id": task, "drive_root": str(child)})
    result = headless.retry_child_task_refs(parent, child, task)
    assert result["child_ref_promotion"]["status"] == "complete"
    assert result["child_ref_promotion"]["pending_refs"] == []


def _unowned_attachment(tmp_path):
    """A captured checkpoint whose contract names an attachment outside every owned root."""
    parent, task = tmp_path / "canonical", "unowned-attachment"
    child = headless.prepare_task_drive(parent, task, "empty")
    outside = tmp_path / "elsewhere" / "screen.png"
    outside.parent.mkdir()
    outside.write_bytes(b"fake-png-bytes")
    row = {"status": "ok", "abs_path": str(outside), "size": outside.stat().st_size, "sha256": "a" * 64}
    captured = _acceptance_checkpoint(child, task, {"attachment_manifest": [row]})
    write_task_result(child, task, "completed", result="answer", artifact_status="ready",
                      review_evidence={"source_refs": [captured]})
    headless.copy_child_task_result(parent, {"id": task, "drive_root": str(child)})
    return parent, child, task, outside


def test_a_deterministic_contract_refusal_is_walked_once_per_unchanged_basis(tmp_path, monkeypatch):
    parent, _child, task, outside = _unowned_attachment(tmp_path)
    counts = _observe(monkeypatch)
    assert observability.retry_pending_child_ref_promotions(parent)["pending"] == [task]
    refused = [row for row in load_task_result(parent, task)["child_ref_promotion"]["pending_refs"]
               if row.get("failure_kind") == "invalid_source"]
    assert refused and refused[0]["watched_paths"] == [str(outside)]
    assert refused[0]["reason"].endswith("review attachment has no captured owner-bound source")
    first = dict(counts)
    for _ in range(2):  # the next runs, in any process: no walk, no read, no event
        report = observability.retry_pending_child_ref_promotions(parent)
        assert report["unchanged"] == [task] and not report["retried"] and counts == first
    outside.write_bytes(b"different bytes")  # a file the verdict read changed: walk it again
    assert observability.retry_pending_child_ref_promotions(parent)["retried"] == [task]
    assert counts["reads"] > first["reads"]
    assert observability.retry_pending_child_ref_promotions(parent)["unchanged"] == [task]


def test_a_transient_failure_of_the_same_source_stays_retryable(tmp_path, monkeypatch):
    parent, _child, task, _outside = _unowned_attachment(tmp_path)
    real, failed = source_retention.RetentionWalk._source, []

    def flaky(self, node, raw):
        if not failed:
            failed.append(node["ref"]["path"])
            raise OSError("transient retention write failure")
        return real(self, node, raw)

    monkeypatch.setattr(source_retention.RetentionWalk, "_source", flaky)
    assert observability.retry_pending_child_ref_promotions(parent)["pending"] == [task]
    pending = load_task_result(parent, task)["child_ref_promotion"]["pending_refs"]
    assert failed and not any(row.get("failure_kind") for row in pending), "no negative observation"
    # Transient I/O is walked again; the real refusal behind it is then remembered.
    assert observability.retry_pending_child_ref_promotions(parent)["retried"] == [task]
    assert observability.retry_pending_child_ref_promotions(parent)["unchanged"] == [task]
