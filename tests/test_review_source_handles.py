"""Review/completion handles retain exact bytes through publication and child cleanup."""
import hashlib
import json
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import artifacts, review_projection
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
    assert copied["child_ref_promotion"]["promoted_source_handle_count"] == 1
    assert remove_subagent_task_drive(parent, "source") is True
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
        assert copied["child_ref_promotion"]["status"] == "incomplete"
        assert remove_subagent_task_drive(parent, "source") is False
        assert child.exists()
    copied = copy_child_task_result(parent, {"id": "source", "drive_root": str(child)})
    assert copied["child_ref_promotion"]["status"] == "complete"
    assert remove_subagent_task_drive(parent, "source") is True
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
        assert client.get(url).content == b"user result"
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
