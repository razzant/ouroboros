"""Review/completion handles retain exact bytes through publication and child cleanup."""
from ouroboros.headless import retry_child_task_refs
import hashlib
import json
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import artifacts, review_projection
from ouroboros.gateway import task_archive
from ouroboros.gateway.tasks import api_task_artifact
from ouroboros.headless import copy_child_task_result, prepare_task_drive, remove_subagent_task_drive
from ouroboros.task_results import write_task_result
from tests.test_acceptance_publication import _context, _run


def _field(ref, source):
    if source == "review":
        return {"review_projection": {"panels": [{"surface": "task_acceptance", "applied_source_ref": ref}]}}
    return {"completion_observations": {"source_ref": ref, "source_status": "available"}}


@pytest.mark.parametrize("source", ["review", "completion"])
def test_child_source_closure_survives_real_cleanup(tmp_path, source):
    parent = tmp_path / "canonical"
    child = prepare_task_drive(parent, "source", "empty")
    raw = b'{"full":"retained evidence"}'
    ref = artifacts.store_actor_source_bytes(child, "source", category="context_checkpoints",
                                             source_id=source, data=raw, extension="json")
    write_task_result(child, "source", "completed", **_field(ref, source))
    copied = copy_child_task_result(parent, {"id": "source", "drive_root": str(child)})
    copied = retry_child_task_refs(parent, child, "source")
    assert copied["child_ref_promotion"]["promoted_source_handle_count"] == 1
    assert remove_subagent_task_drive(parent, "source", live=lambda _task: False) is True
    assert not child.exists()
    assert artifacts.read_actor_source_bytes(parent, "source", ref) == raw
    assert artifacts.collect_task_artifact_records(parent, "source") == []


def test_failed_completion_promotion_retains_child_until_retry(tmp_path, monkeypatch):
    parent = tmp_path / "canonical"
    child = prepare_task_drive(parent, "source", "empty")
    raw = b'{"full":"survives failed copy"}'
    ref = artifacts.store_actor_source_bytes(child, "source", category="context_checkpoints",
                                             source_id="completion", data=raw, extension="json")
    write_task_result(child, "source", "completed", **_field(ref, "completion"))
    with monkeypatch.context() as patch:
        patch.setattr(artifacts, "store_actor_source_bytes", lambda *_a, **_k: (_ for _ in ()).throw(OSError("copy failed")))
        copied = copy_child_task_result(parent, {"id": "source", "drive_root": str(child)})
        copied = retry_child_task_refs(parent, child, "source")
        assert copied["child_ref_promotion"]["status"] == "incomplete"
        assert remove_subagent_task_drive(parent, "source", live=lambda _task: False) is False
        assert child.exists()
    copied = copy_child_task_result(parent, {"id": "source", "drive_root": str(child)})
    copied = retry_child_task_refs(parent, child, "source")
    assert copied["child_ref_promotion"]["status"] == "complete"
    assert remove_subagent_task_drive(parent, "source", live=lambda _task: False) is True
    assert artifacts.read_actor_source_bytes(parent, "source", ref) == raw


def test_unchanged_review_source_is_write_once(tmp_path, monkeypatch):
    ctx = _context(tmp_path)
    trace = {"review_runs": [_run()]}
    review_projection.publish_acceptance_checkpoint(ctx, trace)
    ref = trace["review_runs"][0]["applied_source_ref"]
    path = artifacts.task_artifact_dir_path(tmp_path, "applied") / ref["path"]
    stamp = path.stat().st_mtime_ns
    monkeypatch.setattr(artifacts, "write_bytes_atomic", lambda *_a, **_k: pytest.fail("same source rewritten"))
    review_projection.publish_acceptance_checkpoint(ctx, trace)
    assert trace["review_runs"][0]["applied_source_ref"] == ref
    assert path.stat().st_mtime_ns == stamp
    assert not (path.parents[2] / artifacts._ARTIFACT_MANIFEST).exists()


@pytest.mark.serial
def test_source_download_is_bound_and_distinct_from_same_named_user_file(tmp_path):
    ctx = _context(tmp_path)
    trace = {"review_runs": [_run()]}
    review_projection.publish_acceptance_checkpoint(ctx, trace)
    ref = trace["review_runs"][0]["applied_source_ref"]
    name = Path(ref["path"]).name
    artifacts.store_task_artifact_bytes(tmp_path, "applied", name, b"user result", kind="user_file")
    app = Starlette(routes=[Route("/api/tasks/{task_id}/artifacts/{name}", api_task_artifact)])
    app.state.drive_root = tmp_path
    with TestClient(app) as client:
        url = f"/api/tasks/applied/artifacts/{name}"
        artifact = client.get(url)
        if task_archive.CONFINED:
            assert artifact.status_code == 200 and artifact.content == b"user result"
        else:
            assert artifact.status_code == 503
            assert artifact.json()["reason_code"] == "artifact_unavailable"
        source = client.get(url, params={"source": ref["path"]})
        assert source.status_code == 200
        assert hashlib.sha256(source.content).hexdigest() == ref["sha256"]
        assert len(json.loads(source.content)["actors"][0]["parsed"]["findings"]) == 80
        assert client.get(url, params={"source": "source_handles/context_checkpoints/not-published.json"}).status_code == 404
        (artifacts.task_artifact_dir_path(tmp_path, "applied") / ref["path"]).write_bytes(b"corrupt")
        assert client.get(url, params={"source": ref["path"]}).status_code == 404


@pytest.mark.parametrize("panels", [True, 7, {"legacy": "unknown"}])
def test_unavailable_review_projection_does_not_hide_valid_completion_source(tmp_path, panels):
    raw = b'{"delivery_results": []}'
    ref = artifacts.store_actor_source_bytes(tmp_path, "source", category="context_checkpoints",
                                             source_id="completion", data=raw, extension="json")
    result = {"task_id": "source", "review_projection": {"panels": panels},
              "completion_observations": {"source_ref": ref}}
    assert artifacts.read_task_result_source_bytes(tmp_path, result, Path(ref["path"]).name, ref["path"]) == raw


def _nested_acceptance_sources(child):
    result = artifacts.store_actor_source_bytes(
        child, "source", category="tool_results", source_id="command-output",
        data=b"complete output beyond the preview", extension="txt",
    )
    trajectory = artifacts.persist_tool_trajectory_source(child, "source", [{
        "tool": "run_command", "result": "preview", "result_partial": True,
        "result_source_ref": result,
    }])
    raw = json.dumps({"authority": "host_root", "request": {
        "surface": "task_acceptance", "evidence": {"tool_trajectory_source_ref": trajectory},
    }}).encode()
    checkpoint = artifacts.store_actor_source_bytes(
        child, "source", category="context_checkpoints", source_id="acceptance",
        data=raw, extension="json",
    )
    write_task_result(child, "source", "completed", **_field(checkpoint, "review"))
    return checkpoint, trajectory, result


@pytest.mark.parametrize("outer_already_copied", [False, True])
def test_nested_acceptance_sources_survive_copy_back_and_cleanup(tmp_path, outer_already_copied):
    parent = tmp_path / "canonical"
    child = prepare_task_drive(parent, "source", "empty")
    refs = _nested_acceptance_sources(child)
    before = [artifacts.read_actor_source_bytes(child, "source", ref) for ref in refs]
    if outer_already_copied:
        artifacts.store_actor_source_bytes(parent, "source", category="context_checkpoints",
                                           source_id="acceptance", data=before[0], extension="json")
    for _ in range(2):
        copied = copy_child_task_result(parent, {"id": "source", "drive_root": str(child)})
        copied = retry_child_task_refs(parent, child, "source")
        assert copied["child_ref_promotion"]["status"] == "complete"
        assert copied["child_ref_promotion"]["promoted_source_handle_count"] == 3
        assert copied["review_projection"]["panels"][0]["applied_source_ref"] == refs[0]
    assert remove_subagent_task_drive(parent, "source", live=lambda _task: False) is True
    assert not child.exists()
    assert [artifacts.read_actor_source_bytes(parent, "source", ref) for ref in refs] == before
    assert artifacts.collect_task_artifact_records(parent, "source") == []
    # The trajectory producer preserves the complete original handle contract.
    assert refs[1]["size"] == len(before[1])
    assert refs[1]["sha256"] == hashlib.sha256(before[1]).hexdigest()
    assert refs[1]["read"]["arguments"]["path"] == refs[1]["path"]


def test_nested_copy_failure_holds_child_and_rechecks_existing_checkpoint(tmp_path, monkeypatch):
    parent = tmp_path / "canonical"
    child = prepare_task_drive(parent, "source", "empty")
    checkpoint, trajectory, result = _nested_acceptance_sources(child)
    store = artifacts.store_actor_source_bytes

    def fail_output(*args, **kwargs):
        if kwargs.get("source_id") == "command-output":
            raise OSError("output copy failed")
        return store(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(artifacts, "store_actor_source_bytes", fail_output)
        copied = copy_child_task_result(parent, {"id": "source", "drive_root": str(child)})
        copied = retry_child_task_refs(parent, child, "source")
        promotion = copied["child_ref_promotion"]
        assert promotion["status"] == "incomplete"
        assert promotion["pending_refs"][0]["path"] == str(
            artifacts.task_artifact_dir_path(child, "source") / result["path"])
        assert artifacts.read_actor_source_bytes(parent, "source", checkpoint)
        assert artifacts.read_actor_source_bytes(parent, "source", trajectory)
        assert remove_subagent_task_drive(parent, "source", live=lambda _task: False) is False
    copied = copy_child_task_result(parent, {"id": "source", "drive_root": str(child)})
    copied = retry_child_task_refs(parent, child, "source")
    assert copied["child_ref_promotion"]["status"] == "complete"
    assert copied["child_ref_promotion"]["pending_refs"] == []
    assert remove_subagent_task_drive(parent, "source", live=lambda _task: False) is True
    assert artifacts.read_actor_source_bytes(parent, "source", result) == b"complete output beyond the preview"


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("copy_failure", [False, True])
@pytest.mark.parametrize("canonical_checkpoint", [False, True], ids=["child_checkpoint", "canonical_checkpoint"])
def test_nested_trajectory_promotes_existing_call_blobs_without_crawling_prose(tmp_path, monkeypatch, selected, copy_failure, canonical_checkpoint):
    from ouroboros.observability import persist_call, promote_child_task_refs, retry_pending_child_ref_promotions
    from ouroboros.review_evidence import build_task_acceptance_evidence
    from ouroboros.review_evidence_refs import acceptance_evidence_ref_vocabulary, resolve_criteria_evidence_refs
    from ouroboros.task_results import load_task_result
    from ouroboros.utils import sanitize_tool_args_for_log

    parent = tmp_path / "canonical"
    child = prepare_task_drive(parent, "source", "empty")
    args = {"cmd": "x" * (4000 if selected else 100), "literal": str(child / "not-a-dependency.json")}
    call = {"tool": "run_command", "tool_call_id": "call", "args": args, "result": "ok"}
    trace = persist_call(child, task_id="source", call_id="call", call_type="tool", payload=call)
    ctx = _context(child)
    ctx.task_id = "source"
    checkpoint_root = parent if canonical_checkpoint else child
    ctx.task_metadata = {"budget_drive_root": str(checkpoint_root)}
    evidence = build_task_acceptance_evidence(
        ctx, drive_root=child, task_id="source",
        llm_trace={"tool_calls": [{**call, "args": sanitize_tool_args_for_log("run_command", args), "trace_ref": trace}]},
        agent_evidence={"tool_trajectory_indices": [0], "corpus_sha256": "f" * 64} if selected else None,
    )
    trajectory = evidence["tool_trajectory_source_ref"]
    assert "corpus_sha256" not in trajectory  # The model cannot author the host handle.
    section = "tool_trajectory_selected" if selected else "tool_trajectory"
    citation = evidence[section][0]["ref"]
    assert acceptance_evidence_ref_vocabulary(evidence)[citation] == "tool_record"
    criteria = [{"criterion": "command", "status": "supported", "evidence_refs": [citation]}]
    run = _run()
    run["request"].update(task_id="source", evidence=evidence)
    run["actors"][0]["parsed"]["criteria_used"] = criteria
    review_trace = {"review_runs": [run]}
    review_projection.publish_acceptance_checkpoint(ctx, review_trace)
    checkpoint = review_trace["review_runs"][0]["applied_source_ref"]
    original_checkpoint = artifacts.read_actor_source_bytes(checkpoint_root, "source", checkpoint)
    write_task_result(child, "source", "completed",
                      review_projection=load_task_result(checkpoint_root, "source")["review_projection"])
    task = {"id": "source", "drive_root": str(child)}
    if copy_failure:
        store = artifacts.store_actor_source_bytes

        def fail_corpus(*args, **kwargs):
            if kwargs.get("source_id") == "acceptance_tool_trajectory":
                raise OSError("corpus copy failed")
            return store(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(artifacts, "store_actor_source_bytes", fail_corpus)
            copied = copy_child_task_result(parent, task)
            copied = retry_child_task_refs(parent, child, task["id"])
            assert copied["child_ref_promotion"]["status"] == "incomplete"
            assert copied["child_ref_promotion"]["pending_refs"]
            assert remove_subagent_task_drive(parent, "source", live=lambda _task: False) is False
        assert retry_pending_child_ref_promotions(parent)["completed"] == ["source"]
    first_ref = None
    for _ in range(2):
        copied = copy_child_task_result(parent, task)
        copied = retry_child_task_refs(parent, child, task["id"])
        assert copied["child_ref_promotion"]["status"] == "complete"
        ref = copied["review_projection"]["panels"][0]["applied_source_ref"]
        first_ref = first_ref or ref
        assert ref == first_ref
        assert "corpus_sha256" not in ref  # Checkpoints keep their own byte identity.
    assert artifacts.read_actor_source_bytes(checkpoint_root, "source", checkpoint) == original_checkpoint
    assert remove_subagent_task_drive(parent, "source", live=lambda _task: False) is True
    assert not child.exists()
    # Rebase the already promoted source again, then repeat after the child is
    # gone. Neither a second physical digest nor idempotent copying renames rows.
    other = tmp_path / "another-canonical"
    for destination, source_root in ((parent, child), (other, parent), (other, parent)):
        copied, state = promote_child_task_refs(destination, source_root, "source", copied)
        assert state["status"] == "complete" and not state["pending_refs"]
        ref = copied["review_projection"]["panels"][0]["applied_source_ref"]
        saved = json.loads(artifacts.read_actor_source_bytes(destination, "source", ref))
        evidence = saved["request"]["evidence"]
        promoted = evidence["tool_trajectory_source_ref"]
        assert promoted["sha256"] == trajectory["sha256"]
        assert "corpus_sha256" not in promoted  # Exact captured bytes need no rebased citation identity.
        assert promoted["artifact_ref"].startswith(f"artifact_store:{promoted['path']}#chars=")
        raw = artifacts.read_actor_source_bytes(destination, "source", promoted)
        assert hashlib.sha256(raw).hexdigest() == promoted["sha256"]
        with pytest.raises(ValueError, match="sha256 verification"):
            artifacts.read_actor_source_bytes(destination, "source", {**promoted, "sha256": "0" * 64})
        assert saved["actors"][0]["parsed"]["criteria_used"] == criteria
        assert evidence[section][0]["ref"] == citation
        vocabulary = acceptance_evidence_ref_vocabulary(evidence)
        assert vocabulary[citation] == "tool_record"
        assert resolve_criteria_evidence_refs(criteria, vocabulary) == []
        recovered, complete, issue = artifacts.materialize_tool_args_source(destination, json.loads(raw)[0])
        assert complete and not issue and recovered == args
        assert artifacts.collect_task_artifact_records(destination, "source") == []


@pytest.mark.parametrize("outer_already_copied", [False, True])
@pytest.mark.parametrize("loss", ["missing", "corrupt", "legacy_ref"])
def test_nested_unavailable_source_is_disclosed_without_reconstructing_preview(tmp_path, loss, outer_already_copied):
    parent = tmp_path / "canonical"
    child = prepare_task_drive(parent, "source", "empty")
    checkpoint, trajectory, result = _nested_acceptance_sources(child)
    source_path = artifacts.task_artifact_dir_path(child, "source") / result["path"]
    if loss == "missing":
        source_path.unlink()
    elif loss == "corrupt":
        source_path.write_bytes(b"corrupt")
    else:
        # The historical producer dropped these fields; the filename is not a
        # substitute for the missing producer-issued verification contract.
        legacy = {k: v for k, v in trajectory.items() if k not in {"read", "size", "sha256"}}
        raw = json.dumps({"request": {"surface": "task_acceptance", "evidence": {
            "tool_trajectory_source_ref": legacy}}}).encode()
        checkpoint = artifacts.store_actor_source_bytes(
            child, "source", category="context_checkpoints", source_id="acceptance",
            data=raw, extension="json")
        write_task_result(child, "source", "completed", **_field(checkpoint, "review"))
    if outer_already_copied:
        artifacts.store_actor_source_bytes(
            parent, "source", category="context_checkpoints", source_id="acceptance",
            data=artifacts.read_actor_source_bytes(child, "source", checkpoint), extension="json")
        # Same host publication on both roots must still publish its physical
        # missing-source disclosure after the normal merge selects canonical.
        projection = {"panels": [{"surface": "task_acceptance", "panel_id": "p",
                                   "publication_revision": 1, "applied_source_ref": checkpoint}]}
        write_task_result(parent, "source", "running", review_projection=projection)
        write_task_result(child, "source", "completed", review_projection=projection)
    copied = copy_child_task_result(parent, {"id": "source", "drive_root": str(child)})
    copied = retry_child_task_refs(parent, child, "source")
    assert copied["child_ref_promotion"]["unavailable_refs"]
    promoted = copied["review_projection"]["panels"][0]["applied_source_ref"]
    assert promoted == checkpoint  # Captured evidence is immutable, even when a dependency is unavailable.
    evidence = json.loads(artifacts.read_actor_source_bytes(parent, "source", promoted))["request"]["evidence"]
    dependency = evidence["tool_trajectory_source_ref"]
    if loss != "legacy_ref":
        dependency = json.loads(artifacts.read_actor_source_bytes(parent, "source", dependency))[0]["result_source_ref"]
    with pytest.raises((OSError, ValueError)):
        artifacts.read_actor_source_bytes(parent, "source", dependency)
    expected = {"missing": "source_missing", "corrupt": "digest_mismatch", "legacy_ref": "invalid_ref"}[loss]
    assert any(row["reason"] == expected for row in copied["child_ref_promotion"]["unavailable_refs"])
    assert not (artifacts.task_artifact_dir_path(parent, "source") / result["path"]).exists()


def test_copyback_prepares_bulk_artifacts_and_selected_review_refs_outside_lock(tmp_path, monkeypatch):
    from ouroboros import headless, observability

    parent = tmp_path / "canonical"
    child = prepare_task_drive(parent, "source", "empty")
    _nested_acceptance_sources(child)
    artifacts.store_task_artifact_bytes(child, "source", "report.txt", b"actual deliverable")
    write_task_result(child, "source", "completed", artifacts=artifacts.collect_task_artifact_records(child, "source"))
    lock = parent / "task_results" / "source.json.lock"
    bulk_copy = headless._copy_child_artifacts_to_parent
    promote = observability._promote_task_source_ref
    observed = []

    def copy_bulk(*args, **kwargs):
        assert not lock.exists(), "bulk artifact copying entered the result lock"
        observed.append("bulk")
        return bulk_copy(*args, **kwargs)

    def promote_review(*args, **kwargs):
        assert not lock.exists(), "CURRENT review reference I/O entered the result lock"
        observed.append("review")
        return promote(*args, **kwargs)

    monkeypatch.setattr(headless, "_copy_child_artifacts_to_parent", copy_bulk)
    monkeypatch.setattr(observability, "_promote_task_source_ref", promote_review)
    copied = copy_child_task_result(parent, {"id": "source", "drive_root": str(child)})
    copied = retry_child_task_refs(parent, child, "source")
    assert copied["child_ref_promotion"]["status"] == "complete"
    assert observed[0] == "bulk" and "review" in observed
    assert remove_subagent_task_drive(parent, "source", live=lambda _task: False) is True
    assert (artifacts.task_artifact_dir_path(parent, "source") / "report.txt").read_bytes() == b"actual deliverable"


@pytest.mark.parametrize('delivery', ['manifest', 'blob'])
def test_response_capture_does_not_launder_nested_source_authority(tmp_path, monkeypatch, delivery):
    """A verified capture proves bytes, not that model-authored locators are host edges."""
    from ouroboros import observability

    parent = tmp_path / 'canonical'
    child = prepare_task_drive(parent, 'source', 'empty')
    unselected = artifacts.store_actor_source_bytes(child, 'source', category='tool_results',
        source_id='unselected', data=b'UNREAD', extension='txt')
    payload = {'source_ref': unselected, 'trace_refs': {'response': unselected},
               'messages': [{'role': 'tool', 'tool_call_id': 'unselected',
                             'content': 'FULL_RESULT_SOURCE_JSON=' + json.dumps(unselected)}],
               'message': {'source_ref': unselected}, 'body': {'dialogue_source_ref': unselected}}
    ref = (observability.persist_call(child, task_id='source', call_id='response',
        call_type='llm_response', payload=payload)['manifest_ref'] if delivery == 'manifest'
        else observability.write_blob(child, payload))
    artifacts.store_task_artifact_bytes(child, 'source', 'report.txt', b'deliverable')
    artifact = artifacts.collect_task_artifact_records(child, 'source')[0]
    write_task_result(child, 'source', 'completed', trace_refs={'response': ref},
        artifacts=[{**artifact, 'source_ref': unselected}],
        review_evidence={'agent_supplied': payload, 'unknown': payload,
                         'receipt': unselected, 'plan_review_state': payload,
                         '__provenance__': {'plan_review_state': 'agent_supplied'}})
    read = artifacts.read_actor_source_bytes

    def guarded(root, owner, source):
        assert source != unselected, 'model/unknown evidence acquired source authority'
        return read(root, owner, source)

    monkeypatch.setattr(artifacts, 'read_actor_source_bytes', guarded)
    copied = copy_child_task_result(parent, {'id': 'source', 'drive_root': str(child)})
    copied = retry_child_task_refs(parent, child, 'source')
    assert not copied['child_ref_promotion']['unavailable_refs']
    assert copied['review_evidence']['agent_supplied'] == payload
    assert copied['review_evidence']['receipt'] == unselected
    assert remove_subagent_task_drive(parent, 'source', live=lambda _: False)
    promoted = copied['trace_refs']['response']
    if delivery == 'manifest':
        promoted = observability.read_call_manifest_ref(parent, promoted, task_id='source')['full_payload_ref']
    assert observability.read_blob_ref(parent, promoted) == payload
    assert not (artifacts.task_artifact_dir_path(parent, 'source') / unselected['path']).exists()


def _record_path_reads(monkeypatch, forbidden):
    """Record every pathlib open and refuse ``forbidden`` before its bytes are read.

    ``Path.open`` sits under ``read_bytes`` on every supported Python; patching
    ``io.open`` missed 3.10, whose pathlib accessor bound ``io.open`` at import.
    Promotion turns the refusal into an unavailable ref, so the record, not the
    raise, is the negative proof.
    """
    opened, real_open = [], Path.open

    def guard(path, *args, **kwargs):
        opened.append(path.resolve())
        assert opened[-1] != forbidden, 'foreign bytes read before confinement check'
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guard)
    return opened


@pytest.mark.parametrize('fault', ['sibling', 'symlink', 'unknown'])
def test_trace_response_rejects_foreign_paths_before_read_and_keeps_unknown_custody(tmp_path, monkeypatch, fault):
    from ouroboros import observability

    parent = tmp_path / 'canonical'
    child = prepare_task_drive(parent, 'source', 'empty')
    sibling = prepare_task_drive(parent, 'sibling', 'empty')
    foreign = observability.persist_call(sibling, task_id='source', call_id='response',
                                         call_type='llm_response', payload={'message': 'UNREAD'})['manifest_ref']
    forbidden = Path(foreign['path']).resolve()
    ref = dict(foreign)
    if fault == 'symlink':
        local = child / 'observability/calls/source/response.json'
        local.parent.mkdir(parents=True)
        local.symlink_to(forbidden)
        ref['path'] = str(local)
    elif fault == 'unknown':
        ref = observability.persist_call(child, task_id='source', call_id='response',
            call_type='llm_response', payload={'message': 'retained answer'})['manifest_ref']
    write_task_result(child, 'source', 'completed', trace_refs={'response': ref}, result='retained answer')
    own = Path(ref['path']).resolve()
    opened = _record_path_reads(monkeypatch, forbidden)
    if fault == 'unknown':
        with monkeypatch.context() as fail:
            fail.setattr(observability, 'write_call_manifest', lambda *a, **k: (_ for _ in ()).throw(OSError('copy unknown')))
            copied = copy_child_task_result(parent, {'id': 'source', 'drive_root': str(child)})
            copied = retry_child_task_refs(parent, child, 'source')
            assert copied['child_ref_promotion']['pending_refs']
            assert not remove_subagent_task_drive(parent, 'source', live=lambda _: False)
            assert child.exists() and Path(ref['path']).exists()
        copied = copy_child_task_result(parent, {'id': 'source', 'drive_root': str(child)})
        copied = retry_child_task_refs(parent, child, 'source')
        assert not copied['child_ref_promotion']['pending_refs']
        # Even verified publication cannot collect a drive with unknown liveness.
        assert not remove_subagent_task_drive(parent, 'source', live=lambda _: None)
        assert remove_subagent_task_drive(parent, 'source', live=lambda _: False)
    else:
        copied = copy_child_task_result(parent, {'id': 'source', 'drive_root': str(child)})
        copied = retry_child_task_refs(parent, child, 'source')
        assert copied['trace_refs']['response']['availability'] == 'unavailable'
        assert copied['child_ref_promotion']['unavailable_refs']
        assert not list((parent / 'observability').rglob('response.json'))
    assert forbidden not in opened
    if fault == 'unknown':
        assert own in opened  # the copy's own manifest read passed through the guard
    # Positive control: the same guard intercepts the manifest reader on this very file.
    with pytest.raises(AssertionError, match='foreign bytes'):
        observability.read_call_manifest_ref(sibling, foreign, task_id='source')
    assert opened[-1] == forbidden


@pytest.mark.parametrize('link', [False, True], ids=['sibling', 'symlink'])
def test_plan_dialogue_promotion_never_reads_a_foreign_source(tmp_path, monkeypatch, link):
    parent = tmp_path / 'canonical'
    child = prepare_task_drive(parent, 'source', 'empty')
    ref = artifacts.store_actor_source_bytes(child, 'sibling', category='context_checkpoints',
        source_id='plan-dialogue-1', data=b'SIBLING UNREAD', extension='jsonl')
    sibling_ref = json.loads(json.dumps(ref))
    foreign = (artifacts.task_artifact_dir_path(child, 'sibling') / ref['path']).resolve()
    if link:
        target = artifacts.task_artifact_dir_path(child, 'source') / ref['path']
        target.parent.mkdir(parents=True)
        target.symlink_to(foreign)
    else:
        ref['path'] = '../sibling/' + ref['path']
        ref['read']['arguments']['path'] = ref['path']
    write_task_result(child, 'source', 'completed', plan_review_state={
        'schema_version': 2, 'waves': [{'request_fingerprint': 'f' * 64, 'dialogue_source_ref': ref}]})
    opened = _record_path_reads(monkeypatch, foreign)
    copied = copy_child_task_result(parent, {'id': 'source', 'drive_root': str(child)})
    copied = retry_child_task_refs(parent, child, 'source')
    assert copied['plan_review_state']['waves'][0]['dialogue_source_ref']['availability'] == 'unavailable'
    assert copied['child_ref_promotion']['unavailable_refs']
    assert not (artifacts.task_artifact_dir_path(parent, 'sibling') / ref['path']).exists()
    assert foreign not in opened
    # Positive control: the same guard intercepts the source reader on this very file.
    with pytest.raises(AssertionError, match='foreign bytes'):
        artifacts.read_actor_source_bytes(child, 'sibling', sibling_ref)
    assert opened[-1] == foreign


@pytest.mark.parametrize("missing", [False, True])
def test_submitted_plan_source_survives_promotion_and_gates_child_cleanup(tmp_path, missing):
    from ouroboros.task_results import record_plan_review_attempt, load_plan_review_state
    from ouroboros.tools.plan_author_history import current_submitted_plan
    parent = tmp_path / "canonical"
    child = prepare_task_drive(parent, "submitted", "empty")
    proposed = {"spec": {"goal": "Keep exact submitted material"}, "plan_prose": "Current unreviewed proposal"}
    state = record_plan_review_attempt(child, "submitted", fingerprint="b" * 64, submitted_subject=proposed)
    ref = state["current_attempt"]["submitted_subject"]
    path = artifacts.task_artifact_dir_path(child, "submitted") / ref["path"]
    raw = path.read_bytes()
    write_task_result(child, "submitted", "completed", result="No reviewer was dispatched")
    if missing:
        path.unlink()
        copy_child_task_result(parent, {"id": "submitted", "drive_root": str(child)})
        assert not remove_subagent_task_drive(parent, "submitted", live=lambda _: False)
        assert child.exists()
        path.write_bytes(raw)
    copy_child_task_result(parent, {"id": "submitted", "drive_root": str(child)})
    promoted = retry_child_task_refs(parent, child, "submitted")
    assert promoted["child_ref_promotion"]["status"] == "complete"
    assert remove_subagent_task_drive(parent, "submitted", live=lambda _: False)
    assert not child.exists()
    assert artifacts.read_actor_source_bytes(parent, "submitted", ref) == raw
    current = current_submitted_plan(parent, "submitted", load_plan_review_state(parent, "submitted"))
    assert current["spec"] == proposed["spec"] and current["plan_prose"] == proposed["plan_prose"]
