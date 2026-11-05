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
from tests._store_shared import result_signature


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
