"""File completion preserves sources and publishes CURRENT refs without bulk locks."""

import copy
import gzip
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from ouroboros import headless, observability as obs
from ouroboros.artifacts import read_actor_source_bytes, store_actor_source_bytes
from ouroboros.outcomes import collect_trace_refs
from ouroboros.store_import import ensure_current_store
from ouroboros.task_results import load_task_result, write_task_result
from tests._store_shared import read_result, result_signature


def _child(tmp_path, task_id="copyback"):
    parent = tmp_path / "data"
    parent.mkdir()
    child = headless.prepare_task_drive(parent, task_id, "empty")
    return parent, child, {"id": task_id, "drive_root": str(child), "_attempt": 1}


def _call(root, task_id="copyback", call_id="call"):
    return obs.persist_call(root, task_id=task_id, call_id=call_id,
                            call_type="llm_response", payload={"answer": "exact answer"})


def _review(ref, revision=1, headline="reviewed"):
    return {"panels": [{"surface": "task_acceptance", "panel_id": "panel",
                        "panel_index": 0, "task_attempt": 1, "publication_revision": revision,
                        "status": "PASS", "headline": headline, "source_ref": ref}]}


def _store(root, task_id, ref, **fields):
    # The real trace producer names one call through its LLM-response and tool
    # edges, then loop_outcome repeats the response: with a review panel, four
    # occurrences of one call, so a copy must still do each file operation once.
    usage = {"llm_call_refs": [{"llm_call_id": ref["call_id"], "response_ref": ref}]}
    tool = {"tool_calls": [{"trace_ref": {"call_id": ref["call_id"], "manifest_ref": ref}}]}
    return write_task_result(root, task_id, "completed", result="done", artifact_status="ready",
                             trace_refs=collect_trace_refs(usage, tool),
                             loop_outcome={"trace_refs": collect_trace_refs(usage, {})}, **fields)


def _trace_occurrences(result):
    trace, nested = result["trace_refs"], result["loop_outcome"]["trace_refs"]
    return [trace["llm_call_refs"][0]["response_ref"], trace["tool_call_refs"][0]["manifest_ref"],
            nested["llm_call_refs"][0]["response_ref"]]


@pytest.mark.parametrize("alias", [False, True])
def test_same_store_keeps_original_manifest_and_reads_old_refs_after_alias_cleanup(tmp_path, monkeypatch, alias):
    parent = tmp_path / "data"
    parent.mkdir()
    child = tmp_path / "old-alias" if alias else parent
    if alias:
        child.symlink_to(parent, target_is_directory=True)
    # Reproduce already-stored legacy lexical paths, before root canonicalization.
    with monkeypatch.context() as legacy:
        def lexical_root(root):
            path = Path(root) / "observability"
            path.mkdir(parents=True, exist_ok=True)
            return path
        legacy.setattr(obs, "_observability_root", lexical_root)
        trace = _call(child)
    original_ref = copy.deepcopy(trace["manifest_ref"])
    native_path = Path(original_ref["path"]).resolve()
    native_bytes = native_path.read_bytes()
    _store(child, "copyback", original_ref, review_projection=_review(original_ref))

    copied = headless.copy_child_task_result(parent, {"id": "copyback", "drive_root": str(child)})

    assert copied["child_ref_promotion"]["status"] == "complete"
    assert copied["child_ref_promotion"]["promoted_ref_count"] == 0
    assert native_path.read_bytes() == native_bytes
    assert "promoted_call_manifest" not in json.loads(native_bytes)
    refs = [original_ref, *_trace_occurrences(copied), copied["review_projection"]["panels"][0]["source_ref"]]
    if alias:
        child.unlink()
    for ref in refs:
        manifest = obs.read_call_manifest_ref(parent, ref, task_id="copyback")
        assert obs.read_blob_ref(parent, manifest["full_payload_ref"]) == {"answer": "exact answer"}
        assert ref["sha256"] == original_ref["sha256"]
    assert list(native_path.parent.glob("*.json")) == [native_path]


def test_shared_observability_directory_is_same_store(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    ref = _call(source)["manifest_ref"]
    original = Path(ref["path"]).read_bytes()
    (target / "observability").symlink_to(source / "observability", target_is_directory=True)
    monkeypatch.setattr(obs, "write_call_manifest", lambda *a, **k: pytest.fail("same store rewritten"))
    promoted = obs.promote_call_manifest_ref(source, target, ref, task_id="copyback")
    assert promoted["sha256"] == ref["sha256"]
    assert Path(promoted["path"]).read_bytes() == original


@pytest.mark.parametrize("defect", ["sha", "encoding", "existing_outside"])
def test_missing_locator_fallback_does_not_replace_invalid_blob_evidence(tmp_path, monkeypatch, defect):
    root = tmp_path / "data"
    ref = obs.write_blob(root, {"answer": "exact"})
    old = {**ref, "path": str(tmp_path / "retired" / "observability" / "blobs" / Path(ref["path"]).name)}
    assert obs.read_blob_ref(root, old) == {"answer": "exact"}
    if defect == "sha":
        old["sha256"] = "0" * 64
    elif defect == "size":
        old["size"] += 1
    elif defect == "encoding":
        old["encoding"] = "plain"
    else:
        outside = Path(old["path"])
        outside.parent.mkdir(parents=True)
        outside.write_bytes(b"unrelated bytes must never be opened")
        real_open = gzip.open
        def open_inside(path, *args, **kwargs):
            assert Path(path) != outside
            return real_open(path, *args, **kwargs)
        monkeypatch.setattr(gzip, "open", open_inside)
        assert obs.read_blob_ref(root, old) == {"answer": "exact"}
        return
    with pytest.raises((ValueError, OSError)):
        obs.read_blob_ref(root, old)


def test_manifest_fallback_keeps_exact_task_call_and_digest_checks(tmp_path):
    root = tmp_path / "data"
    ref = _call(root)["manifest_ref"]
    old = {**ref, "path": str(tmp_path / "retired" / "observability" / "calls" / "copyback" / "call.json")}
    assert obs.read_call_manifest_ref(root, old, task_id="copyback")["call_id"] == "call"
    for bad, task in [({**old, "sha256": "0" * 64}, "copyback"),
                      ({**old, "call_id": "other"}, "copyback"), (old, "other")]:
        with pytest.raises((ValueError, OSError)):
            obs.read_call_manifest_ref(root, bad, task_id=task)
    Path(ref["path"]).write_text("{}")
    with pytest.raises(ValueError, match="sha256"):
        obs.read_call_manifest_ref(root, old, task_id="copyback")


def test_distinct_retention_reuses_shared_sources_and_revalidates_inventory(tmp_path, monkeypatch):
    from ouroboros import artifacts
    parent, child, task = _child(tmp_path)
    ref = _call(child)["manifest_ref"]
    source = store_actor_source_bytes(child, task["id"], category="tool_results", source_id="source",
                                     data=b"exact retained source", extension="txt")
    _store(child, task["id"], ref, review_evidence={"sources": [source, source]},
           review_projection=_review(ref))
    writes = Counter()
    real_manifest, real_source = obs.write_call_manifest, artifacts.store_actor_source_bytes
    def manifest(*args, **kwargs):
        writes["manifest"] += 1
        return real_manifest(*args, **kwargs)
    def source_store(*args, **kwargs):
        writes["source"] += 1
        return real_source(*args, **kwargs)
    monkeypatch.setattr(obs, "write_call_manifest", manifest)
    monkeypatch.setattr(artifacts, "store_actor_source_bytes", source_store)
    headless.copy_child_task_result(parent, task)
    copied = headless.retry_child_task_refs(parent, child, task["id"])
    assert copied["child_ref_promotion"]["status"] == "complete"
    assert writes == {"manifest": 1, "source": 1}
    promoted = _trace_occurrences(copied)[0]
    assert obs.read_call_manifest_ref(parent, promoted, task_id=task["id"])["promoted_call_manifest"] is True
    assert read_actor_source_bytes(parent, task["id"], source) == b"exact retained source"
    Path(ref["path"]).write_text("{}", encoding="utf-8")
    # A later copy uses no previous operation's verification. The valid retained
    # version remains readable, but the newly corrupt inventory cannot certify GC.
    headless.copy_child_task_result(parent, task)
    second = headless.retry_child_task_refs(parent, child, task["id"])
    assert second["child_ref_promotion"]["status"] == "incomplete"
    assert obs.read_call_manifest_ref(parent, ref, task_id=task["id"])["call_id"] == "call"


@pytest.mark.parametrize(("change_source", "retry"), [(False, False), (True, True)])
def test_current_review_writer_runs_during_bulk_io_and_wins(tmp_path, monkeypatch, retry, change_source):
    parent, child, task = _child(tmp_path)
    old = _call(child, call_id="old")["manifest_ref"]
    new = _call(parent, call_id="new")["manifest_ref"]
    _store(child, task["id"], old, review_projection=_review(old))
    if retry:
        headless.copy_child_task_result(parent, task)
    else:
        write_task_result(parent, task["id"], "completed", review_projection=_review(old))
    entered, release = Event(), Event()
    original = obs._rewrite_child_ref_tree
    waits = []
    def rewrite(value, *args, **kwargs):
        # Only CURRENT review preparation: ordinary trace copying already finished.
        if isinstance(value, dict) and "panels" in value and not waits:
            waits.append(True)
            entered.set()
            assert release.wait(5), "concurrent CURRENT writer was blocked by bulk I/O"
        return original(value, *args, **kwargs)
    monkeypatch.setattr(obs, "_rewrite_child_ref_tree", rewrite)
    if not retry:
        from ouroboros import history_retention
        prepare = history_retention.prepare_result_retention
        def preparing(*args, **kwargs):
            value = prepare(*args, **kwargs)
            entered.set()
            assert release.wait(5), "concurrent CURRENT writer was blocked by deliverable I/O"
            return value
        monkeypatch.setattr(history_retention, "prepare_result_retention", preparing)
    operation = (lambda: headless.retry_child_task_refs(parent, child, task["id"])) if retry else (
        lambda: headless.copy_child_task_result(parent, task))
    with ThreadPoolExecutor(max_workers=2) as pool:
        copying = pool.submit(operation)
        assert entered.wait(5)
        current_review = _review(new if change_source else old, 2 if change_source else 1, "CURRENT headline")
        def concurrent_write():
            # The publication owner may enrich metadata without changing its ref identity.
            return write_task_result(parent, task["id"], "completed", result="CURRENT result",
                                     accounted_upper_bound_usd=17, review_projection=current_review,
                                     _field_projector=lambda _current, fields: {**fields, "review_projection": current_review})
        updating = pool.submit(concurrent_write)
        try:
            updating.result(timeout=2)
        finally:
            release.set()
        result = copying.result(timeout=5)
    assert result["accounted_upper_bound_usd"] == 17
    if retry:
        assert result["result"] == "CURRENT result"
    panel = result["review_projection"]["panels"][0]
    assert panel["headline"] == "CURRENT headline"
    assert panel["publication_revision"] == (2 if change_source else 1)
    resolved = obs.read_call_manifest_ref(parent, panel["source_ref"], task_id=task["id"])
    assert resolved["call_id"] == ("new" if change_source else "old")
    assert bool(result["child_ref_promotion"]["pending_refs"]) is (not retry)


@pytest.mark.parametrize("source_status", [None, "running"])
def test_helper_never_finalizes_without_terminal_child_source(tmp_path, monkeypatch, source_status):
    parent, child, task = _child(tmp_path)
    # Post-task can already have written a canonical checkpoint without adopting the body.
    write_task_result(parent, task["id"], "completed", root_phase_checkpoint={"post_task_synthesis": "completed"})
    if source_status:
        write_task_result(child, task["id"], source_status, result="not done")
    monkeypatch.setattr(headless, "finalize_task_artifacts", lambda *a: pytest.fail("invented terminal source"))
    report = headless.prepare_terminal_task_files(parent, task)
    assert report["task_id"] == task["id"] and report["error"]
    assert report["terminal_source_present"] is False
    assert "result" not in load_task_result(parent, task["id"])


@pytest.mark.parametrize("recorded_placement", [True, False])
def test_copyback_stamps_its_read_drive_without_borrowing_replica_placement(tmp_path, recorded_placement):
    from ouroboros.task_status import effective_task_result

    parent, child, task = _child(tmp_path)
    tid = task["id"]
    placement = {"drive_root": str(child), "child_drive_root": str(child), "budget_drive_root": str(parent)}
    admitted = placement if recorded_placement else {}
    # A legacy row can have its placement only in metadata. Neither spelling may be replaced by copyback.
    write_task_result(parent, tid, "running", **admitted, metadata={"child_drive_root": str(child)})
    older = {"drive_root": str(parent), "child_drive_root": str(parent),
             "budget_drive_root": str(child), "headless_child_drive_root": str(parent)}
    write_task_result(child, tid, "completed", result="finished notes",
                      **older, metadata={**older, "summary": "notes from the worker"})

    def check(result, expected_placement):
        assert result["result"] == "finished notes"
        assert result["headless_child_drive_root"] == str(child)
        assert {key: result[key] for key in placement if key in result} == expected_placement
        assert result["metadata"] == {"child_drive_root": str(child), "summary": "notes from the worker"}

    copied = headless.copy_child_task_result(parent, task)
    check(copied, admitted)
    assert headless.terminal_task_files_ready(parent, task, copied)
    # The effective read still supplies a display-only child_drive_root on legacy rows.
    check(effective_task_result(parent, copied, materialize_artifacts=False), {"child_drive_root": str(child), **admitted})
    check(headless.retry_child_task_refs(parent, child, tid), admitted)


def test_missing_terminal_source_does_not_stamp_running_result_with_artifact_failure(tmp_path, monkeypatch):
    parent, child, task = _child(tmp_path)
    write_task_result(parent, task["id"], "running", result="still executing")
    monkeypatch.setattr(headless, "finalize_task_artifacts", lambda *a: pytest.fail("invented terminal source"))
    report = headless.prepare_terminal_task_files(parent, task)
    current = load_task_result(parent, task["id"], strict=True)
    assert report["terminal_source_present"] is False
    assert report["error"]
    assert current["status"] == "running"
    assert "artifact_status" not in current and "artifact_error" not in current


def test_helper_retains_terminal_child_when_canonical_write_fails(tmp_path, monkeypatch):
    parent, child, task = _child(tmp_path)
    _store(child, task["id"], _call(child)["manifest_ref"])
    source = result_signature(child, task['id'])
    monkeypatch.setattr(headless, "write_task_result", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    report = headless.prepare_terminal_task_files(parent, task)
    assert report == {"task_id": task["id"], "result": None, "error": "OSError: disk full", "terminal_source_present": True}
    assert result_signature(child, task['id']) == source
    assert load_task_result(parent, task["id"]) is None


def test_helper_finalizes_once_and_recovery_keeps_current_body(tmp_path, monkeypatch):
    parent, child, task = _child(tmp_path)
    task["workspace_root"] = str(tmp_path)
    monkeypatch.setattr(headless, "write_workspace_patch_artifacts", lambda *a, **k: ([], {"status": "ready_no_changes"}))
    _store(child, task["id"], _call(child)["manifest_ref"])
    write_task_result(child, task["id"], "completed", artifact_status="pending")
    first = headless.prepare_terminal_task_files(parent, task)
    assert first["error"] == ""
    assert first["result"]["artifact_finalized_at"]
    write_task_result(parent, task["id"], "completed", result="CURRENT authority")
    monkeypatch.setattr(headless, "finalize_task_artifacts", lambda *a: pytest.fail("recaptured finalized artifacts"))
    second = headless.prepare_terminal_task_files(parent, task)
    assert second["error"] == "" and second["result"]["result"] == "CURRENT authority"


def test_helper_readonly_skips_artifact_capture(tmp_path, monkeypatch):
    parent, child, task = _child(tmp_path)
    _store(child, task["id"], _call(child)["manifest_ref"])
    task.update(delegation_role="subagent", task_constraint={"mode": "local_readonly_subagent"})
    monkeypatch.setattr(headless, "finalize_task_artifacts", lambda *a: pytest.fail("readonly artifact capture"))
    assert headless.prepare_terminal_task_files(parent, task)["error"] == ""


def test_retry_refuses_malformed_current_without_rebuilding_from_child(tmp_path):
    parent, child, task = _child(tmp_path)
    _store(child, task["id"], _call(child)["manifest_ref"])
    path = parent / "task_results" / f"{task['id']}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text("{", encoding="utf-8")
    ensure_current_store(parent)  # Boot quarantines the malformed legacy authority.
    child_signature = result_signature(child, task["id"])
    with pytest.raises(ValueError, match="authority is unreadable or invalid"):
        headless.retry_child_task_refs(parent, child, task["id"])
    assert load_task_result(parent, task["id"]) is None
    assert result_signature(child, task["id"]) == child_signature
    assert (path.parent / "quarantine" / path.name).read_bytes() == b"{"


def test_native_model_send_reverse_reader_accepts_retired_alias(tmp_path, monkeypatch):
    from ouroboros import model_send_seal as seal
    root = tmp_path / "data"
    root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with monkeypatch.context() as legacy:
        legacy.setattr(obs, "_observability_root", lambda p: Path(p) / "observability")
        persisted = seal.persist_physical_candidate(alias, task_id="copyback", attempt_id="attempt",
                                                    candidate={"messages": []}, candidate_facts={})
    ref = persisted["manifest_ref"]
    obs.promote_call_manifest_ref(alias, root, ref, task_id="copyback")
    alias.unlink()
    facts = []
    report = {"seals": 0, "sealed_attempts": 0, "orphan_seals": 0, "unlogged_attempts": 0}
    seal._reconcile_seal_directions(root, {"attempt": {"state": "settled", "task_id": "copyback",
                                                    "candidate_manifest_ref": ref}}, facts.append, report,
                                    seal._seal_manifest_paths(root, 20))
    assert report["seals"] == report["sealed_attempts"] == 1
    assert report["orphan_seals"] == report["unlogged_attempts"] == 0
    assert facts == []


def test_early_post_task_and_failed_copy_never_certify_ready(tmp_path, monkeypatch):
    parent, child, task = _child(tmp_path)
    _store(child, task["id"], _call(child)["manifest_ref"])
    early = write_task_result(parent, task["id"], "completed", root_phase_checkpoint={"post_task_synthesis": "completed"})
    monkeypatch.setattr(headless, "write_task_result", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    report = headless.prepare_terminal_task_files(parent, task)
    current = load_task_result(parent, task["id"], strict=True)
    assert current == early
    assert report["terminal_source_present"] is True and report["error"]
    assert not headless.terminal_task_files_ready(parent, task, current)


def test_unreadable_required_source_is_unknown_not_absent(tmp_path, monkeypatch):
    parent, child, task = _child(tmp_path)
    monkeypatch.setattr(headless, "load_task_result", lambda *a, **k: (_ for _ in ()).throw(OSError("read failed")))
    report = headless.prepare_terminal_task_files(parent, task)
    assert report["terminal_source_present"] is None and report["error"]


def test_current_readiness_uses_existing_adoption_and_artifact_facts(tmp_path):
    parent, child, task = _child(tmp_path)
    _store(child, task["id"], _call(child)["manifest_ref"])
    result = headless.prepare_terminal_task_files(parent, task)["result"]
    assert headless.terminal_task_files_ready(parent, task, result)
    pending = {**result, "child_ref_promotion": {**result["child_ref_promotion"], "status": "incomplete", "pending_refs": [{}]}}
    assert headless.terminal_task_files_ready(parent, task, pending)
    assert not headless.terminal_task_files_ready(parent, task, {**pending, "headless_child_drive_root": str(parent)})
    assert not headless.terminal_task_files_ready(parent, task, {**pending, "child_ref_promotion": {"schema_version": 2}})
    workspace = {**task, "workspace_root": str(tmp_path / "workspace")}
    assert not headless.terminal_task_files_ready(parent, workspace, {**result, "artifact_status": "finalizing"})
    assert headless.terminal_task_files_ready(parent, workspace, {**result, "artifact_status": "failed"})
    assert headless.terminal_task_files_ready(parent, task, {"task_id": task["id"], "status": "cancelled"})
    assert headless.terminal_task_files_ready(parent, {"id": task["id"]}, {"task_id": task["id"], "status": "failed"})


def test_helper_canonical_cancellation_precedes_child_retry(tmp_path, monkeypatch):
    parent, child, task = _child(tmp_path)
    cancelled = write_task_result(parent, task["id"], "cancelled", result="stopped",
                                  headless_child_drive_root=str(child), child_ref_promotion={
                                      "schema_version": 1, "status": "incomplete", "pending_refs": [{}]})
    monkeypatch.setattr(headless, "retry_child_task_refs", lambda *a: pytest.fail("cancelled child retry"))
    monkeypatch.setattr(headless, "copy_child_task_result", lambda *a: pytest.fail("cancelled child copy"))
    report = headless.prepare_terminal_task_files(parent, task)
    assert report["terminal_source_present"] is True and report["error"] == ""
    assert report["result"] == cancelled
    assert headless.terminal_task_files_ready(parent, task, cancelled)


_OWN = {'result': 'Attempt 2 stopped by the host.', 'reason_code': 'provider_unavailable',
        'terminal_origin': 'host_notice', 'loop_outcome': {'usage': {'total_rounds': 0}}}
_COPY = {'result': 'Answer held in the child drive.', 'terminal_origin': 'model_final',
         'loop_outcome': {'usage': {'total_rounds': 3}}}


def _ending(row):
    # What a reader takes of an attempt's ending from its stored row: status, attempt, reason, author, counter.
    row = row or {}
    return {'status': row.get('status'), 'attempt': row.get('task_attempt'), 'reason': row.get('reason_code'),
            'origin': row.get('terminal_origin'), 'result': row.get('result'),
            'rounds': ((row.get('loop_outcome') or {}).get('usage') or {}).get('total_rounds')}


@pytest.mark.parametrize(('canonical', 'replica_attempt', 'lands'), [
    pytest.param(('failed', 2), 1, False, id='older-beside-failed'),
    pytest.param(('running', 2), 1, False, id='older-beside-running'),
    pytest.param(('interrupted', 2), 1, False, id='older-beside-interrupted'),
    pytest.param(('running', 2), 2, True, id='same-attempt-finishes'),
    pytest.param(('interrupted', 2), 2, True, id='interrupted-own-finish'),
    pytest.param(('completed', 2), 2, True, id='completed-mirror-enriches'),
    pytest.param(('running', None), None, True, id='unnumbered-legacy-pair'),
    pytest.param(None, 1, True, id='empty-parent-first-copy'),
    pytest.param(('failed', 1), 0, False, id='zero-beside-failed-one'),
    pytest.param(('failed', 0), 1, False, id='one-beside-failed-zero'),
    pytest.param(('running', 1), 0, False, id='zero-beside-running-one'),
    pytest.param(('failed', 0), None, False, id='unnumbered-beside-attempt-zero'),
    pytest.param(('failed', 1), 1, False, id='same-ordinal-other-ending'),
    pytest.param(('running', 0), 0, True, id='attempt-zero-finishes'),
])
def test_a_late_copy_never_lends_its_ending_to_another_attempt(tmp_path, canonical, replica_attempt, lands):
    # Another numbered attempt's result — older or newer, and zero is the real first attempt, never "unset" or one —
    # changes neither a failed attempt's own ending (an own ordinal ended differently keeps its own stored zero) nor
    # an open attempt's status. The attempt's own finish (its own ordinal, zero included), a wholly unnumbered
    # legacy pair and an empty parent adopt theirs. Completed mirrors take the worker's compatible enrichment.
    from ouroboros.task_status import load_effective_task_result

    parent, child, task = _child(tmp_path)
    tid, own = task['id'], {}
    if canonical:
        status, attempt = canonical
        if status == 'failed':
            own = copy.deepcopy(_OWN)
        elif status == 'completed':
            own = {'result': 'First completed mirror.', 'terminal_origin': 'model_final',
                   'loop_outcome': {'usage': {'total_rounds': 1}}}
        write_task_result(parent, tid, status, child_drive_root=str(child), _terminal_observed=status == 'failed',
                          ts='2026-11-21T11:00:00+00:00', **own, **({} if attempt is None else {'task_attempt': attempt}))
    source = write_task_result(child, tid, 'completed', _terminal_observed=True,
                               ts='2026-11-21T12:00:00+00:00', **copy.deepcopy(_COPY),
                               **({} if replica_attempt is None else {'task_attempt': replica_attempt}))
    expected = _ending({'status': canonical[0], 'task_attempt': canonical[1], **own}) if canonical else _ending(None)
    before = _ending(read_result(parent, tid))
    assert before == expected
    if before['status'] == 'completed':
        effective = load_effective_task_result(parent, tid, materialize_artifacts=False)
        assert (_ending(effective), effective['ts']) == (_ending(source), source['ts'])
    headless.copy_child_task_result(parent, task)
    stored = read_result(parent, tid)
    now = _ending(stored)
    if lands:
        assert stored['ts'] == source['ts'] and 'late_attempt_results' not in stored
        assert (now['status'], now['origin'], now['rounds'], now['result']) == (
            'completed', 'model_final', 3, _COPY['result'])
        assert replica_attempt is None or now['attempt'] == replica_attempt
    else:
        assert now == expected


def test_a_kept_late_result_list_survives_a_matching_replicas_old_copy(tmp_path):
    # The row keeps other attempts' replies as its own canonical custody (``late_attempt_results``). A later
    # matching replica of the winner attempt carries the field's older snapshot — the same stale copy a split
    # worker carries of ``focus`` or ``continuation_narrative`` — and its own different ending for that attempt.
    # Neither the effective read nor the copy-back lets that old copy age or drop the kept list: the old entries
    # stay the identical prefix, the rejected ending is kept whole behind them, and every entry stays addressable.
    from ouroboros.task_status import load_effective_task_result

    parent, _prepared, task = _child(tmp_path)
    tid = task['id']
    write_task_result(parent, tid, 'failed', task_attempt=2, _terminal_observed=True, **copy.deepcopy(_OWN))
    for name, attempt in (('first', 1), ('second', 0)):
        drive = tmp_path / name
        drive.mkdir()
        write_task_result(drive, tid, 'completed', task_attempt=attempt, _terminal_observed=True,
                          result=f'Attempt {attempt} reply.', loop_outcome={'usage': {'total_rounds': attempt + 1}})
        headless.copy_child_task_result(parent, {'id': tid, 'drive_root': str(drive)})
    kept = read_result(parent, tid)['late_attempt_results']
    assert [row['task_attempt'] for row in kept] == [1, 0]  # two distinct physical attempts, one entry each

    stale = tmp_path / 'third'
    stale.mkdir()
    write_task_result(stale, tid, 'completed', task_attempt=2, _terminal_observed=True,
                      late_attempt_results=[kept[0]], result='Winner attempt reply.',
                      loop_outcome={'usage': {'total_rounds': 9}})
    write_task_result(parent, tid, 'failed', child_drive_root=str(stale))  # its own attempt's drive now serves the row
    assert load_effective_task_result(parent, tid, materialize_artifacts=False)['late_attempt_results'] == kept

    headless.copy_child_task_result(parent, {'id': tid, 'drive_root': str(stale)})

    stored = read_result(parent, tid)
    grown = stored['late_attempt_results']
    assert grown[:len(kept)] == kept  # the old copy neither ages nor drops the kept list
    rejected = [row for row in grown[len(kept):] if row.get('task_attempt') == 2]
    assert len(grown) == len(kept) + 1 and len(rejected) == 1  # the replica's own ending joins, once
    assert rejected[0]['status'] == 'completed'
    source = json.loads(read_actor_source_bytes(parent, tid, rejected[0]['source_ref']))
    assert (source['status'], source['result'], source['loop_outcome']) == (
        'completed', 'Winner attempt reply.', {'usage': {'total_rounds': 9}})  # that whole changed result
    assert (stored['status'], stored['result']) == ('failed', _OWN['result'])  # the chosen ending stays the host's
    assert load_effective_task_result(parent, tid, materialize_artifacts=False)['late_attempt_results'] == grown

    headless.copy_child_task_result(parent, {'id': tid, 'drive_root': str(stale)})
    assert read_result(parent, tid)['late_attempt_results'] == grown  # a repeated copy adds no duplicate


def test_a_kept_ending_keeps_its_own_clock_beside_its_workers_late_one(tmp_path):
    # The host already chose attempt 2's ending — its provider failure, at its own clock. That same attempt's
    # worker then supplies a different ending from its drive: the copy-back keeps the chosen words, rounds,
    # origin and reason, and their timestamp too, never the worker's earlier clock.
    parent, child, task = _child(tmp_path)
    tid = task['id']
    write_task_result(parent, tid, 'failed', task_attempt=2, child_drive_root=str(child), _terminal_observed=True,
                      ts='2026-11-21T12:00:00+00:00', **copy.deepcopy(_OWN))
    write_task_result(child, tid, 'completed', task_attempt=2, _terminal_observed=True,
                      ts='2026-11-21T11:00:00+00:00', **copy.deepcopy(_COPY))
    chosen = read_result(parent, tid)
    assert chosen['ts'] == '2026-11-21T12:00:00+00:00'

    headless.copy_child_task_result(parent, task)

    stored = read_result(parent, tid)
    assert (stored['status'], stored['result'], stored['ts']) == (
        'failed', _OWN['result'], '2026-11-21T12:00:00+00:00')
    assert stored['terminal_time'] == chosen['terminal_time']  # the observed end stays the host's fact
    assert _ending(stored)['rounds'] == 0 and _ending(stored)['origin'] == 'host_notice'
    assert stored.get('child_status') == 'completed'


def test_a_kept_ending_keeps_its_clock_when_the_host_settles_mid_copy(tmp_path, monkeypatch):
    # The host chooses attempt 2's ending while that attempt's own worker result is being copied back: the
    # re-prepared write keeps the chosen words, rounds and freshly chosen clock together, never the worker's.
    from ouroboros.task_results import write_task_result as real_write

    parent, child, task = _child(tmp_path)
    tid = task['id']
    write_task_result(parent, tid, 'running', task_attempt=2, child_drive_root=str(child))
    write_task_result(child, tid, 'completed', task_attempt=2, _terminal_observed=True,
                      ts='2026-11-21T11:00:00+00:00', **copy.deepcopy(_COPY))
    raced = []

    def race(root, task_id, status, **fields):
        if not raced:  # the host's ending lands between the copy's basis read and its commit
            raced.append(True)
            real_write(parent, tid, 'failed', _terminal_observed=True, ts='2026-11-21T12:00:00+00:00',
                       **copy.deepcopy(_OWN))
        return real_write(root, task_id, status, **fields)

    monkeypatch.setattr(headless, 'write_task_result', race)
    headless.copy_child_task_result(parent, task)

    stored = read_result(parent, tid)
    assert (stored['status'], stored['result'], stored['ts']) == (
        'failed', _OWN['result'], '2026-11-21T12:00:00+00:00')
    assert _ending(stored)['rounds'] == 0 and stored.get('child_status') == 'completed'

    kept = [row for row in stored['late_attempt_results'] if row.get('task_attempt') == 2]
    assert len(kept) == 1 and kept[0]['status'] == 'completed'
    original = json.loads(read_actor_source_bytes(parent, tid, kept[0]['source_ref']))
    assert (original['status'], original['result'], original['ts'], original['terminal_origin']) == (
        'completed', _COPY['result'], '2026-11-21T11:00:00+00:00', 'model_final')
    assert original['loop_outcome'] == _COPY['loop_outcome']  # the drive's whole original row
    assert 'child_ref_promotion' not in original  # the original bytes, not this copy's prepared metadata

    headless.copy_child_task_result(parent, task)
    assert read_result(parent, tid)['late_attempt_results'] == stored['late_attempt_results']  # one per digest


def _old_attempt(child, tid, same_name=None):
    # Attempt 1's completed result as it waits in its child drive: its reply, a unique attachment and, when asked,
    # a file named like the winner's. Returns the child's artifact folder.
    from ouroboros import artifacts

    store = artifacts.task_artifact_dir_path(child, tid, create=True)
    files = {'old-only.txt': b'unique old attachment', **({'report.txt': same_name} if same_name else {})}
    for name, data in files.items():
        (store / name).write_bytes(data)
    write_task_result(child, tid, 'completed', task_attempt=1, _terminal_observed=True, **copy.deepcopy(_COPY),
                      artifacts=[artifacts.artifact_record(store / name) for name in files])
    return store


def _listed(parent, tid):
    # The artifacts a real read of the task shows, by name.
    from ouroboros.task_status import load_effective_task_result

    return {row.get('relpath') or row['name']: row for row in load_effective_task_result(parent, tid)['artifacts']}


def test_a_late_old_copy_keeps_the_winners_files_exact_and_its_own_unique_ones_readable(tmp_path):
    # Attempt 2 won with its own report. Attempt 1's old result arrives late with a report of the same name, a file
    # nobody else has and its reply: the winner's ending, row and bytes stay exact; the old reply and attachment stay
    # readable through the real reads (in the child drive, or wherever later custody places them).
    from ouroboros import artifacts

    parent, child, task = _child(tmp_path)
    tid = task['id']
    won = artifacts.task_artifact_dir_path(parent, tid, create=True)
    (won / 'report.txt').write_bytes(b'winner report')
    winner = artifacts.artifact_record(won / 'report.txt')
    write_task_result(parent, tid, 'failed', task_attempt=2, child_drive_root=str(child), _terminal_observed=True,
                      artifacts=[winner], **copy.deepcopy(_OWN))
    _old_attempt(child, tid, same_name=b'old report of attempt 1')
    before = _ending(read_result(parent, tid))
    assert (before['status'], before['rounds'], before['result']) == ('failed', 0, _OWN['result'])  # a real winner

    headless.copy_child_task_result(parent, task)

    stored = read_result(parent, tid)
    assert _ending(stored) == before
    assert (won / 'report.txt').read_bytes() == b'winner report'
    row = next(item for item in stored['artifacts'] if Path(item['path']).resolve() == (won / 'report.txt').resolve())
    assert row['sha256'] == winner['sha256']  # the winner's row still names the winner's bytes
    assert load_task_result(child, tid)['result'] == _COPY['result']  # the old reply is kept whole
    assert Path(_listed(parent, tid)['old-only.txt']['path']).read_bytes() == b'unique old attachment'


@pytest.mark.parametrize('retain_fails', [False, True])
def test_a_late_reply_survives_its_drives_settlement_in_canonical_custody(tmp_path, monkeypatch, retain_fails):
    # The reply lives in canonical custody, not in the folder: after copy-back and the settlement that removes the
    # source drive (no longer needed), the row's ``late_attempt_results`` source_ref still answers with the old
    # reply's exact bytes, and the old attempt's unique attachment keeps its own file beside the winner's different
    # same-named bytes. A retention that fails certifies nothing: no kept entry, readiness answers False, the drive
    # stays unadopted and the winner's bytes untouched.
    from ouroboros import artifacts, task_custody

    parent, child, task = _child(tmp_path)
    tid = task['id']
    won = artifacts.task_artifact_dir_path(parent, tid, create=True)
    (won / 'report.txt').write_bytes(b'winner report')
    winner = artifacts.artifact_record(won / 'report.txt')
    write_task_result(parent, tid, 'failed', task_attempt=2, child_drive_root=str(child), _terminal_observed=True,
                      artifacts=[winner], **copy.deepcopy(_OWN))
    _old_attempt(child, tid, same_name=b'old report of attempt 1')

    def settle():
        return task_custody.settle_child_drive(parent, tid, child, live=lambda _task: False)

    if retain_fails:
        from ouroboros import history_retention

        def refuse(*_args, **_kwargs):
            raise OSError('no room for the late copy')

        monkeypatch.setattr(history_retention, 'retain_late_result', refuse)
        with pytest.raises(headless.LateResultRetentionFailed, match='no room for the late copy'):
            headless.copy_child_task_result(parent, task)
        stored = read_result(parent, tid)
        assert 'late_attempt_results' not in stored
        assert _ending(stored) == _ending({'status': 'failed', 'task_attempt': 2, **copy.deepcopy(_OWN)})
        assert not headless.terminal_task_files_ready(parent, task, stored)
        assert settle()['reason'] == 'child_result_unadopted' and child.is_dir()
        assert (won / 'report.txt').read_bytes() == b'winner report'
        return

    headless.copy_child_task_result(parent, task)
    entry = read_result(parent, tid)['late_attempt_results'][0]
    assert (entry['task_attempt'], entry['status']) == (1, 'completed')
    assert settle()['reason'] == 'child_refs_pending'  # refs first, exactly as the settlement pass demands
    headless.retry_child_task_refs(parent, child, tid)
    assert settle()['status'] == 'removed'
    assert not child.exists()  # the source folder is no longer needed

    stored = read_result(parent, tid)
    assert _ending(stored) == _ending({'status': 'failed', 'task_attempt': 2, **copy.deepcopy(_OWN)})
    reply = json.loads(read_actor_source_bytes(parent, tid, stored['late_attempt_results'][0]['source_ref']))
    assert reply['result'] == _COPY['result'] and reply['task_attempt'] == 1
    assert reply['loop_outcome'] == _COPY['loop_outcome']  # the exact old reply, from canonical custody
    assert (won / 'report.txt').read_bytes() == b'winner report'  # never replaced by the older same-named file
    copies = {row['name']: Path(row['path']).read_bytes() for row in stored['artifacts']}
    assert copies['report.txt'] == b'winner report'
    assert copies['old-only.txt'] == b'unique old attachment'
    assert b'old report of attempt 1' in copies.values()  # its own file, published beside the winner's


@pytest.mark.parametrize('won', ['stop', 'next-attempt', 'unchanged', 'first-result'])
def test_a_late_file_error_marks_only_the_attempt_it_was_preparing(tmp_path, monkeypatch, won):
    # The copy's own I/O fails after the strict reads. A Stop or the next attempt that took the row meanwhile keeps
    # its own outcome and gets no file failure of the older attempt's preparation; an unchanged matching attempt gets
    # the real error, and a first result meets no row at all. The old source and its attachment stay in every case.
    from ouroboros import history_retention

    parent, child, task = _child(tmp_path)
    tid = task['id']
    if won != 'first-result':
        write_task_result(parent, tid, 'running', task_attempt=1, child_drive_root=str(child))
    store = _old_attempt(child, tid)

    def fail_copy(*_args, **_kwargs):
        if won == 'stop':
            write_task_result(parent, tid, 'cancelled', result='Stopped by the owner.')
        elif won == 'next-attempt':
            write_task_result(parent, tid, 'running', task_attempt=2, result='Attempt 2 is running.')
        raise OSError('disk full while copying')

    monkeypatch.setattr(history_retention, 'prepare_result_retention', fail_copy)
    report = headless.prepare_terminal_task_files(parent, task)
    current = load_task_result(parent, tid) or {}

    if won in ('stop', 'next-attempt'):
        own = {'stop': ('cancelled', 1, 'Stopped by the owner.'),
               'next-attempt': ('running', 2, 'Attempt 2 is running.')}[won]
        assert (current['status'], current.get('task_attempt'), current.get('result')) == own
        assert current.get('artifact_status') != 'failed' and 'artifact_error' not in current
    elif won == 'unchanged':
        assert 'disk full while copying' in report['error']
        assert current['artifact_status'] == 'failed' and 'disk full while copying' in current['artifact_error']
    else:
        assert 'disk full while copying' in report['error'] and not current
    assert load_task_result(child, tid)['result'] == _COPY['result']  # the source stays whole
    assert (store / 'old-only.txt').read_bytes() == b'unique old attachment'
    if current:
        assert Path(_listed(parent, tid)['old-only.txt']['path']).read_bytes() == b'unique old attachment'


def test_a_same_attempt_conflicting_failure_keeps_the_host_ending_and_the_whole_copy_retrievable(tmp_path):
    # One physical attempt, two failed ends: the canonical row holds the host's observed provider failure
    # (its own words, reason, host_notice origin, zero rounds and clock), the same attempt's drive answers
    # with a genuinely different failed end — another reason, counter and closing sentence. The matching
    # ordinal is not an adoption: copy-back keeps the selected canonical ending whole, the drive's full
    # rejected row stays independently retrievable beside it (digest-deduped custody), and the effective
    # read — the same document a reloaded card and a Continue of this attempt read — agrees with the row.
    from ouroboros.task_status import load_effective_task_result

    parent, child, task = _child(tmp_path)
    tid = task['id']
    write_task_result(parent, tid, 'failed', task_attempt=2, child_drive_root=str(child),
                      _terminal_observed=True, ts='2026-11-21T12:00:00+00:00', **copy.deepcopy(_OWN))
    source = write_task_result(child, tid, 'failed', task_attempt=2, _terminal_observed=True,
                               ts='2026-11-21T11:00:00+00:00', reason_code='model_refusal',
                               terminal_origin='model_final', wake_summary='the model stopped itself',
                               error='Worker-only diagnostic.', final_answer='Worker-only answer.',
                               terminal_provider_notice='Worker-only notice.',
                               result='Attempt 2 stopped by the model.', loop_outcome={'usage': {'total_rounds': 2}})
    chosen = read_result(parent, tid)
    absent = {'error', 'final_answer', 'terminal_provider_notice', 'wake_summary'}
    assert not absent & chosen.keys()
    before = load_effective_task_result(parent, tid, materialize_artifacts=False)
    assert _ending(before) == _ending(chosen) and not absent & before.keys()

    headless.copy_child_task_result(parent, task)

    stored = read_result(parent, tid)
    assert not absent & stored.keys()
    assert _ending(stored) == _ending({'status': 'failed', 'task_attempt': 2, **copy.deepcopy(_OWN)})
    assert stored['ts'] == '2026-11-21T12:00:00+00:00'  # words, reason, counter, origin and clock: one ending
    assert stored['terminal_time'] == chosen['terminal_time']  # the observed end stays the host's fact
    assert 'wake_summary' not in stored  # the rejected closing sentence stays with the rejected copy
    kept = [row for row in stored['late_attempt_results'] if row.get('task_attempt') == 2]
    assert len(kept) == 1 and kept[0]['status'] == 'failed'  # same ordinal, still kept whole
    rejected = json.loads(read_actor_source_bytes(parent, tid, kept[0]['source_ref']))
    assert (rejected['result'], rejected['reason_code'], rejected['terminal_origin']) == (
        'Attempt 2 stopped by the model.', 'model_refusal', 'model_final')
    assert rejected == source  # including the worker-only diagnostic, notice and typed answer

    headless.copy_child_task_result(parent, task)
    assert read_result(parent, tid)['late_attempt_results'] == stored['late_attempt_results']  # one per digest

    effective = load_effective_task_result(parent, tid, materialize_artifacts=False)
    assert _ending(effective) == _ending(stored) and not absent & effective.keys()


def test_a_sparse_canonical_mirror_still_adopts_its_own_attempts_full_failed_ending(tmp_path):
    # The boundary the conflict rule must not cross: a canonical mirror naming attempt 2 but carrying no
    # ending (no words, reason or counter yet) beside that same attempt's complete failed end on its drive.
    # The mirror legitimately takes its own full result: the ending lands whole and nothing is kept as a
    # rejected copy.
    parent, child, task = _child(tmp_path)
    tid = task['id']
    write_task_result(parent, tid, 'running', task_attempt=2, child_drive_root=str(child))
    write_task_result(child, tid, 'failed', task_attempt=2, _terminal_observed=True,
                      reason_code='model_refusal', terminal_origin='model_final',
                      result='Attempt 2 stopped by the model.', loop_outcome={'usage': {'total_rounds': 2}})

    headless.copy_child_task_result(parent, task)

    stored = read_result(parent, tid)
    assert (stored['status'], stored['reason_code'], stored['terminal_origin'], stored['result']) == (
        'failed', 'model_refusal', 'model_final', 'Attempt 2 stopped by the model.')
    assert ((stored['loop_outcome'] or {}).get('usage') or {}).get('total_rounds') == 2
    assert 'late_attempt_results' not in stored  # an adoption of the row's own end, not a rejected copy


def test_two_conflicting_endings_of_one_producer_class_keep_the_host_ending_and_retrieve_the_rejected_copy(tmp_path):
    # One physical attempt, two failed ends from the SAME producer class: the canonical row holds the host's
    # observed ending, the same attempt's drive answers with the same status and the same terminal_origin but
    # a different known reason and different words. Same origin is not agreement: the contradictory pair stays
    # distinct — the selected ending whole with its clock and no same-status child echo, the drive's full
    # rejected row kept retrievable beside it, once per digest.
    parent, child, task = _child(tmp_path)
    tid = task['id']
    write_task_result(parent, tid, 'failed', task_attempt=2, child_drive_root=str(child),
                      _terminal_observed=True, ts='2026-11-21T12:00:00+00:00', **copy.deepcopy(_OWN))
    write_task_result(child, tid, 'failed', task_attempt=2, _terminal_observed=True,
                      ts='2026-11-21T11:00:00+00:00', reason_code='model_refusal',
                      terminal_origin='host_notice', wake_summary='the model stopped itself',
                      result='Attempt 2 stopped by the model.', loop_outcome={'usage': {'total_rounds': 2}})
    chosen = read_result(parent, tid)

    headless.copy_child_task_result(parent, task)

    stored = read_result(parent, tid)
    assert _ending(stored) == _ending({'status': 'failed', 'task_attempt': 2, **copy.deepcopy(_OWN)})
    assert stored['ts'] == '2026-11-21T12:00:00+00:00'  # words, reason, counter, origin and clock: one ending
    assert stored['terminal_time'] == chosen['terminal_time']  # the observed end stays the host's fact
    assert 'wake_summary' not in stored  # the rejected closing sentence stays with the rejected copy
    assert 'child_status' not in stored  # one status: never a same-status echo on the kept end
    kept = [row for row in stored['late_attempt_results'] if row.get('task_attempt') == 2]
    assert len(kept) == 1 and kept[0]['status'] == 'failed'  # same class, still a rejected original
    rejected = json.loads(read_actor_source_bytes(parent, tid, kept[0]['source_ref']))
    assert (rejected['result'], rejected['reason_code'], rejected['terminal_origin']) == (
        'Attempt 2 stopped by the model.', 'model_refusal', 'host_notice')
    assert rejected['loop_outcome'] == {'usage': {'total_rounds': 2}}

    headless.copy_child_task_result(parent, task)
    assert read_result(parent, tid)['late_attempt_results'] == stored['late_attempt_results']  # one per digest


def test_a_compatible_same_origin_replica_enriches_and_invents_no_rejected_copy(tmp_path):
    # The other edge the conflict rule must not cross: the drive answers the settled row with the same status,
    # the same terminal_origin and the same known reason, words and counter, and lends only a field the row
    # lacks. Compatible enrichment IS the identical end: the missing field lands, the chosen ending stays
    # whole, and nothing is kept as a rejected original.
    parent, child, task = _child(tmp_path)
    tid = task['id']
    write_task_result(parent, tid, 'failed', task_attempt=2, child_drive_root=str(child),
                      _terminal_observed=True, ts='2026-11-21T12:00:00+00:00', **copy.deepcopy(_OWN))
    write_task_result(child, tid, 'failed', task_attempt=2, _terminal_observed=True,
                      reason_code=_OWN['reason_code'], terminal_origin=_OWN['terminal_origin'],
                      result=_OWN['result'], loop_outcome=copy.deepcopy(_OWN['loop_outcome']),
                      wake_summary='the checkpoint note the row lacked')

    headless.copy_child_task_result(parent, task)

    stored = read_result(parent, tid)
    assert _ending(stored) == _ending({'status': 'failed', 'task_attempt': 2, **copy.deepcopy(_OWN)})
    assert stored['wake_summary'] == 'the checkpoint note the row lacked'  # the compatible field lands
    assert 'late_attempt_results' not in stored  # enrichment of an identical end invents no rejected copy


def test_a_sparse_host_failure_card_reads_its_own_attempts_full_mirror_before_copyback(tmp_path):
    # The reader half of the shared selector: the settled row holds only the host's observed
    # failure (status, reason, origin and clock, no words), while the same attempt's drive
    # carries that attempt's complete failed end. A sparse same-attempt mirror is no
    # conflict: the effective read a card shows BEFORE copy-back already reads the worker's
    # words, error and clock — the same ending the writer's compatible overlay will store,
    # so the card never changes when the copy-back lands.
    from ouroboros.task_status import load_effective_task_result

    parent, child, task = _child(tmp_path)
    tid = task['id']
    write_task_result(parent, tid, 'failed', task_attempt=2, child_drive_root=str(child),
                      _terminal_observed=True, ts='2026-11-21T12:00:00+00:00',
                      reason_code='provider_unavailable', terminal_origin='host_notice')
    write_task_result(child, tid, 'failed', task_attempt=2, _terminal_observed=True,
                      ts='2026-11-21T11:00:00+00:00', reason_code='provider_unavailable',
                      terminal_origin='host_notice', result='Attempt 2 stopped by the model.',
                      error='Worker-only diagnostic.', loop_outcome={'usage': {'total_rounds': 2}})

    before = load_effective_task_result(parent, tid, materialize_artifacts=False)
    assert (before['status'], before['result'], before['error'], before['ts']) == (
        'failed', 'Attempt 2 stopped by the model.', 'Worker-only diagnostic.',
        '2026-11-21T11:00:00+00:00')

    headless.copy_child_task_result(parent, task)

    stored = read_result(parent, tid)
    assert 'late_attempt_results' not in stored  # an enrichment of the same end, not a rejected copy
    assert (stored['status'], stored['result'], stored['error']) == (
        'failed', 'Attempt 2 stopped by the model.', 'Worker-only diagnostic.')
    after = load_effective_task_result(parent, tid, materialize_artifacts=False)
    assert _ending(after) == _ending(stored)


def _conflicted_child(tmp_path, **borrowed):
    # One child drive whose attempt 2 ended differently from the settled row beside it — the conflict shape every
    # control below uses — plus whatever extra ending fields the case lends (``borrowed``).
    parent, child, task = _child(tmp_path)
    tid = task['id']
    write_task_result(parent, tid, 'failed', task_attempt=2, _terminal_observed=True,
                      ts='2026-11-21T12:00:00+00:00', artifact_status='ready',
                      result='Attempt 2 stopped by the host.', reason_code='provider_unavailable',
                      terminal_origin='host_notice', loop_outcome={'usage': {'total_rounds': 0}})
    write_task_result(child, tid, 'failed', task_attempt=2, _terminal_observed=True,
                      ts='2026-11-21T11:00:00+00:00', result='Attempt 2 stopped by the model.',
                      reason_code='model_refusal', terminal_origin='model_final',
                      loop_outcome={'usage': {'total_rounds': 2}}, **borrowed)
    return parent, child, task, tid


def test_a_failed_late_retention_is_no_failure_of_the_winners_own_files(tmp_path, monkeypatch, caplog):
    # The copy-back owes two separate things: this row's own answer and files, and the exact original of the
    # ending it rejects (``history_retention.retain_late_result``). When only THAT duty fails, the winner's
    # bundle is already ready — the row stamps no artifact failure of these files and keeps its chosen ending
    # whole — and the report names the duty that failed. The row claims no adoption either (the publication
    # that would have carried one did not happen), so the drive stays unadopted and the next attempt repeats
    # the same preparation. A failure of the row's OWN file save still stamps one, which
    # ``test_a_late_file_error_marks_only_the_attempt_it_was_preparing`` holds.
    from ouroboros import history_retention

    parent, child, task, tid = _conflicted_child(tmp_path)
    before = read_result(parent, tid)
    secret = 'synthetic-secret-123456'

    def refuse(*_args, **_kwargs):
        raise OSError(f'no room for the late copy; token={secret}')

    monkeypatch.setattr(history_retention, 'retain_late_result', refuse)
    report = headless.prepare_terminal_task_files(parent, task)

    stored = read_result(parent, tid)
    assert stored == before  # no adoption, archive claim or artifact stamp
    assert 'no room for the late copy' in report['error']
    assert report['error'] in caplog.text and secret not in report['error'] and secret not in caplog.text
    assert report['reason_code'] == 'late_result_retention_failed'
    assert report['terminal_source_present'] is True  # this attempt's own settled row is there
    assert report['result'] is None and not headless.terminal_task_files_ready(parent, task, stored)
    assert stored['artifact_status'] == 'ready'  # the winner's ready bundle is no failure of the copy
    assert 'artifact_error' not in stored
    assert _ending(stored) == _ending(before) and stored['result'] == 'Attempt 2 stopped by the host.'
    assert load_task_result(child, tid)['result'] == 'Attempt 2 stopped by the model.'  # still on its drive


def test_a_kept_late_copy_is_re_asked_of_the_basis_the_publication_writes(tmp_path, monkeypatch):
    # CURRENT changes to a compatible ending of the same status and ordinal while retention is off-lock.
    from ouroboros import history_retention

    parent, child, task, tid = _conflicted_child(tmp_path)
    source = load_task_result(child, tid)
    retain = history_retention.retain_late_result
    archived = []

    def retain_then_enrich(root, task_id, rejected):
        entry = retain(root, task_id, rejected)
        archived.append(entry)
        write_task_result(parent, tid, 'failed', **{
            key: source[key] for key in ('result', 'reason_code', 'terminal_origin')})
        return entry

    monkeypatch.setattr(history_retention, 'retain_late_result', retain_then_enrich)
    headless.copy_child_task_result(parent, task)

    stored = read_result(parent, tid)
    [entry] = archived
    assert json.loads(read_actor_source_bytes(parent, tid, entry['source_ref'])) == source
    assert 'late_attempt_results' not in stored  # CURRENT no longer rejects that ending
    assert stored['child_ref_promotion'] and stored['headless_child_drive_root'] == str(child)
    assert _ending(stored) == _ending(source) and stored['ts'] == source['ts']


def test_an_unobserved_replica_count_never_replaces_the_count_the_row_observed(tmp_path):
    # The compatible half of the counter rule: the drive answers the settled row with the same status, reason,
    # origin and words, so the pair is one ending and the field the row lacks lands. Its round count was never
    # read (``loop_outcome.usage.total_rounds`` is ``None``) — unobserved is no evidence and cannot replace the
    # zero this row observed — while a count the replica did read enriches exactly as any other concrete field.
    parent, child, task = _child(tmp_path)
    tid = task['id']
    same_end = {'result': 'Attempt 2 stopped by the host.', 'reason_code': 'provider_unavailable',
                'terminal_origin': 'host_notice'}
    write_task_result(parent, tid, 'failed', task_attempt=2, _terminal_observed=True,
                      ts='2026-11-21T12:00:00+00:00', loop_outcome={'usage': {'total_rounds': 0}}, **same_end)
    write_task_result(child, tid, 'failed', task_attempt=2, _terminal_observed=True,
                      wake_summary='the copy states the same end', **same_end,
                      loop_outcome={'reason_code': same_end['reason_code'],
                                    'usage': {'total_rounds': None, 'completion_tokens': 12}})

    headless.copy_child_task_result(parent, task)

    unread = read_result(parent, tid)
    assert unread['loop_outcome'] == {'reason_code': same_end['reason_code'],
                                     'usage': {'total_rounds': 0, 'completion_tokens': 12}}
    assert unread['wake_summary'] == 'the copy states the same end'  # its concrete field still lands
    assert 'late_attempt_results' not in unread  # the same ending, never a rejected copy

    write_task_result(child, tid, 'failed', task_attempt=2, _terminal_observed=True, **same_end,
                      loop_outcome={'usage': {'total_rounds': 2}})
    headless.copy_child_task_result(parent, task)
    assert ((read_result(parent, tid).get('loop_outcome') or {}).get('usage') or {}).get('total_rounds') == 2


def test_a_rejected_ending_lends_none_of_the_ending_facts_the_card_reads(tmp_path):
    from ouroboros.task_reads import TaskReads

    borrowed = {'cancel_origin': {'source': 'supervisor_cancel'},
                'schedule_name': 'nightly-report', 'review_overrides': {'count': 1, 'latest': 'PASS'},
                'finalization_recovery': {'answer': 'delivered'}, 'body_adoptions': [{'kind': 'body'}]}
    parent, child, task, tid = _conflicted_child(tmp_path, **borrowed)
    write_task_result(parent, tid, 'failed', result='', child_drive_root=str(child))
    before = read_result(parent, tid)
    original = load_task_result(child, tid)
    assert {key: original[key] for key in borrowed} == borrowed
    assert TaskReads(parent).load(child, tid)['_result_text_present'] is True

    def check_display():
        view = TaskReads(parent).effective(tid)
        assert 'result' not in view and view['_result_text_present'] is False
        assert not (borrowed.keys() & view.keys())

    check_display()
    headless.copy_child_task_result(parent, task)
    check_display()

    stored = read_result(parent, tid)
    assert not (borrowed.keys() & stored.keys()) and '_result_text_present' not in stored
    assert _ending(stored) == _ending(before)
    [kept] = stored['late_attempt_results']
    assert (kept['task_attempt'], kept['status']) == (2, 'failed')
    assert json.loads(read_actor_source_bytes(parent, tid, kept['source_ref'])) == original


def test_a_stale_running_replica_cannot_lend_its_answer_presence_to_an_interrupted_row(tmp_path):
    from ouroboros.task_reads import TaskReads
    from ouroboros.task_status import load_effective_task_result

    parent, child, task = _child(tmp_path)
    tid = task['id']
    write_task_result(parent, tid, 'interrupted', result='', task_attempt=2, child_drive_root=str(child))
    write_task_result(child, tid, 'running', result='Stale draft.', task_attempt=2)
    assert load_effective_task_result(parent, tid, materialize_artifacts=False)['result'] == ''
    reads = TaskReads(parent)
    assert reads.load(child, tid)['_result_text_present'] is True
    view = reads.effective(tid)
    assert tid in reads.replica_results
    assert 'result' not in view and view['_result_text_present'] is False
