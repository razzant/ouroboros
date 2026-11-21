"""Service receipt sources and host artifact projection, at the real task gateway."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import artifacts, outcomes, service_outcomes
from ouroboros.gateway import task_archive
from ouroboros.gateway.tasks import api_task_artifact, api_task_get
from ouroboros.observability import write_blob
from ouroboros.task_results import write_task_result

pytestmark = pytest.mark.skipif(not task_archive.CONFINED, reason="needs directory-relative file opens")


@pytest.fixture
def client(tmp_path):
    app = Starlette(routes=[Route("/api/tasks/{task_id}", api_task_get),
                            Route("/api/tasks/{task_id}/artifacts/{name}", api_task_artifact)])
    app.state.drive_root = tmp_path
    with TestClient(app) as http:
        yield http


def _path(task_id, source):
    return f"/api/tasks/{task_id}/artifacts/{source.rsplit('/', 1)[-1]}"


@pytest.mark.parametrize("placement", ["own", "linked", "wrong-digest"])
def test_service_log_source_is_confined_to_its_task(tmp_path, client, placement):
    raw = b"Safe diagnostic\n"
    digest = hashlib.sha256(raw).hexdigest()
    ref = artifacts.store_actor_source_bytes(tmp_path, "producer", data=raw, category="tool_results",
        source_id=service_outcomes.log_source_id(digest, scope="full"), extension="txt")
    source = ref["path"]
    write_task_result(tmp_path, "producer", "completed")
    write_task_result(tmp_path, "someone-else", "completed")
    shaped = artifacts.task_artifact_dir_path(tmp_path, "producer") / source
    if placement == "linked":
        elsewhere = tmp_path / "unrelated.txt"
        elsewhere.write_bytes(raw)  # matching digest is insufficient if the placement is outside the task
        shaped.unlink()
        try:
            shaped.symlink_to(elsewhere)
        except OSError as exc:
            pytest.skip(f"test platform cannot create a file symlink: {type(exc).__name__}")
    elif placement == "wrong-digest":
        shaped.write_bytes(b"wrong content\n")

    answer = client.get(_path("producer", source), params={"source": source})
    assert answer.status_code == (200 if placement == "own" else 404)
    if placement == "own":
        assert answer.content == raw and answer.headers["x-ouroboros-artifact-sha256"] == digest
        assert client.head(_path("producer", source), params={"source": source}).content == b""
    assert client.get(_path("someone-else", source), params={"source": source}).status_code == 404
    assert client.get("/api/tasks/producer/artifacts/not-the-recorded-name.txt", params={"source": source}).status_code == 400


@pytest.mark.parametrize("carrier", ["inline", "artifact", "tail", "other-task", "bad-archive"])
def test_only_a_task_bound_full_legacy_archive_recovers_a_missing_source(tmp_path, client, monkeypatch, carrier):
    from ouroboros.tools import services

    monkeypatch.setattr(services, "open_service_log_source", lambda *a, **kw: pytest.fail("opened a private log"))
    text = "Archived safety-masked diagnostic\n"
    blob = write_blob(tmp_path, text, kind="txt")
    digest = blob["sha256"]
    ref = artifacts.actor_source_reference(category="tool_results", data=text.encode(), extension="txt",
        source_id=service_outcomes.log_source_id(digest, scope="full"))
    log = {"full_log_ref": blob, "source_ref": ref, "deleted_live_log": True,
           "source_scope": "tail" if carrier == "tail" else "full"}
    task_id = "producer"
    ledger = {"task_id": task_id, "entries": [{"kind": "services_stopped", "services": [
        {"task_id": "elsewhere" if carrier == "other-task" else task_id, "lifecycle": "stopped",
         "log_finalization": log}]}]}
    if carrier == "artifact":  # the inline stub cannot name any individual launch
        retained = outcomes.maybe_write_verification_artifact(tmp_path, task_id, ledger, threshold_chars=0)
        ledger, files = retained["inline"], [retained["artifact"]]
    else:
        files = []
    if carrier == "bad-archive":
        Path(blob["path"]).write_bytes(b"not the retained gzip")
    write_task_result(tmp_path, task_id, "completed", verification_ledger=ledger, artifacts=files)

    source = ref["path"]
    reply = client.get(_path(task_id, source), params={"source": source}, headers={"Range": "bytes=0-7"})
    assert reply.status_code == (206 if carrier in {"inline", "artifact"} else 404)
    if reply.status_code == 206:
        assert reply.content == text.encode()[:8] and reply.headers["x-ouroboros-artifact-sha256"] == digest
    assert not (artifacts.task_artifact_dir_path(tmp_path, task_id) / source).exists(), "GET is not a publisher"


@pytest.mark.parametrize("nested", [False, True], ids=["root", "nested"])
def test_same_named_ordinary_artifact_never_uses_the_ledger_projection(tmp_path, client, nested):
    task_id = "producer"
    store = artifacts.task_artifact_dir_path(tmp_path, task_id, create=True)
    relpath = ("user/" if nested else "") + "verification_ledger.json"
    path = store / relpath
    path.parent.mkdir(exist_ok=True)
    raw = b'{ "entries": [{"kind":"services_stopped","services":[{"log_finalization":{"tail":"ordinary"}}]}] }'
    path.write_bytes(raw)
    write_task_result(tmp_path, task_id, "completed", artifacts=[artifacts.artifact_record(path)])
    answer = client.get(_path(task_id, relpath), params={"relpath": relpath} if nested else {})
    assert answer.status_code == 200 and answer.content == raw
    assert answer.headers["x-ouroboros-artifact-sha256"] == hashlib.sha256(raw).hexdigest()


def test_output_version_download_is_exact_but_not_a_second_result_file(tmp_path, client):
    task_id = "producer"
    store = artifacts.task_artifact_dir_path(tmp_path, task_id, create=True)
    report = store / "report.txt"
    report.write_bytes(b"original")
    ref = artifacts.retain_service_output_version(SimpleNamespace(drive_root=tmp_path, task_id=task_id),
                                                  artifacts.artifact_record(report))
    report.write_bytes(b"current")
    write_task_result(tmp_path, task_id, "completed", artifacts=[artifacts.artifact_record(report)])

    answer = client.get(_path(task_id, ref["path"]), params={"source": ref["path"]})
    assert answer.status_code == 200 and answer.content == b"original"
    assert answer.headers["x-ouroboros-artifact-sha256"] == ref["sha256"]
    detail = client.get(f"/api/tasks/{task_id}")
    assert detail.status_code == 200
    assert [row["name"] for row in detail.json()["artifacts"]] == ["report.txt"]
