"""A delegated run's retained journal range reaches the human through the existing task-file route (#1350).

The record's ``source.ref`` names a content-addressed handle in the task's canonical custody
store. The real ``GET /api/tasks/{id}/artifacts/{name}?source=<path>`` handler serves it through
the confined descent, verified against the digest its name carries, whatever became of the
run's engine journal or the task's own execution drive; forged, mismatched, foreign and
linked paths are refused.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from types import SimpleNamespace
from urllib.parse import quote

import pytest

from ouroboros import artifacts, delegate_activity
from ouroboros.gateway import task_archive
from ouroboros.task_results import write_task_result
from tests.test_delegated_activity import Daemon, _gateway, _harness, _records
from tests.test_task_file_serving import _client

TASK = "kid1350"
confined = pytest.mark.skipif(not task_archive.CONFINED, reason="needs directory-relative no-follow opens")


def _retained(tmp_path):
    """A child task whose execution drive is not its custody store records one observation."""
    data = tmp_path / "data"
    drive = data / "task_drives" / TASK
    drive.mkdir(parents=True)
    write_task_result(data, TASK, "running", child_drive_root=str(drive), delegation_role="subagent",
                      parent_task_id="root1350", root_task_id="root1350")
    long_text = "the whole executor sentence " * 400
    daemon = Daemon([[_harness(1, "message", text=long_text), _harness(2, "tool_call", tool={"name": "Read"})]])
    daemon.revealed = 1
    ctx = SimpleNamespace(task_id=TASK, task_attempt=0, drive_root=drive,
                          task_metadata={"budget_drive_root": str(data)})
    record, = _records(ctx, _gateway(daemon), 2)
    return data, drive, daemon, record, long_text


def _url(task, name, source):
    return f"/api/tasks/{task}/artifacts/{quote(name)}?source={quote(source, safe='')}"


def test_split_child_reads_its_produced_source_through_the_published_tool_address(tmp_path):
    from ouroboros.tools.core import _read_file
    from ouroboros.tools.registry import ToolContext

    data, drive, _daemon, record, long_text = _retained(tmp_path)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=drive, task_id=TASK,
                      task_metadata={"budget_drive_root": str(data)})
    ref = record["source"]["ref"]
    result = _read_file(ctx, ref["path"], root=ref["root"])
    assert long_text in result, result
    assert '"seq": 2' in result

    # Own retained sources stay readable without redirecting writes or another
    # task's store to canonical authority.
    from ouroboros.tool_access import build_resolved_resource_binding
    child_store = artifacts.task_artifact_dir_path(drive, TASK, create=False)
    write = build_resolved_resource_binding(ctx, root="artifact_store", path=ref["path"], operation="write")
    assert write.target_path == child_store / ref["path"]
    own_result = artifacts.store_actor_source_bytes(data, TASK, category="tool_results", source_id="own-result", data=b"OWN_RESULT", extension="txt")
    assert "OWN_RESULT" in _read_file(ctx, own_result["path"], root="artifact_store")
    foreign = ToolContext(repo_dir=tmp_path, drive_root=drive, task_id="other-child",
                          task_metadata={"budget_drive_root": str(data)})
    assert long_text not in _read_file(foreign, ref["path"], root=ref["root"])


def test_split_child_activity_source_rejects_symlink_escape_when_supported(tmp_path):
    from ouroboros.tools.core import _read_file
    from ouroboros.tools.registry import ToolContext

    data, drive, _daemon, _record, _long_text = _retained(tmp_path)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=drive, task_id=TASK,
                      task_metadata={"budget_drive_root": str(data)})
    own_store = artifacts.task_artifact_dir_path(data, TASK, create=False)
    other = artifacts.store_actor_source_bytes(data, TASK, category="tool_results", source_id="private",
                                               data=b"NOT_ACTIVITY", extension="txt")
    linked = own_store / "source_handles" / "delegated_activity" / "linked.txt"
    try:
        linked.symlink_to(own_store / other["path"])
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")
    assert "NOT_ACTIVITY" not in _read_file(ctx, "source_handles/delegated_activity/linked.txt", root="artifact_store")


@confined
def test_the_retained_original_downloads_verified_after_the_engine_and_the_drive_are_gone(tmp_path):
    data, drive, daemon, record, long_text = _retained(tmp_path)
    part = record["parts"][0]
    assert part["truncated"] and part["chars"] == len(long_text), "the row carries a bounded preview"
    ref = record["source"]["ref"]
    name = ref["path"].rsplit("/", 1)[1]
    assert (data / "task_results" / "artifacts" / TASK / ref["path"]).is_file(), "kept in the canonical custody store"
    assert not (drive / "task_results").exists(), "never in the execution drive that settlement prunes"
    client = _client(data)

    response = client.get(_url(TASK, name, ref["path"]))
    assert response.status_code == 200, response.text
    assert response.headers["x-ouroboros-artifact-identity"] == "verified"
    assert response.headers["x-ouroboros-artifact-sha256"] == ref["sha256"] == hashlib.sha256(response.content).hexdigest()
    assert len(response.content) == ref["size"]
    lines = [json.loads(line) for line in response.content.decode().splitlines()]
    assert [line["seq"] for line in lines] == [1, 2] and lines[0]["payload"]["text"] == long_text

    shutil.rmtree(drive)                        # the child drive is settled and pruned
    daemon.steps, daemon.revealed = [[]], 1     # and the engine no longer serves the run's journal
    again = client.get(_url(TASK, name, ref["path"]))
    assert again.status_code == 200 and again.content == response.content


@confined
def test_forged_mismatched_foreign_and_linked_sources_are_refused(tmp_path):
    data, _drive, _daemon, record, _text = _retained(tmp_path)
    ref = record["source"]["ref"]
    folder = data / "task_results" / "artifacts" / TASK / "source_handles" / "delegated_activity"
    name = ref["path"].rsplit("/", 1)[1]
    client = _client(data)

    assert client.get(_url(TASK, "other.jsonl", ref["path"])).status_code == 400, "the path must end in the name"
    forged_name = name.replace(ref["sha256"], "0" * 64)
    shutil.copyfile(folder / name, folder / forged_name)
    forged = client.get(_url(TASK, forged_name, f"source_handles/delegated_activity/{forged_name}"))
    assert forged.status_code == 404 and forged.json()["reason_code"] == "artifact_unverified", \
        "bytes that do not hash to the name's digest never leave"
    linked_name = name.replace(ref["sha256"], "1" * 64)
    (folder / linked_name).symlink_to(folder / name)
    assert client.get(_url(TASK, linked_name, f"source_handles/delegated_activity/{linked_name}")).status_code == 404
    for source in (f"source_handles/delegated_activity/../delegated_activity/{name}",
                   f"source_handles/tool_results/{name}", f"/{ref['path']}"):
        assert client.get(_url(TASK, name, source)).status_code == 404, source
    write_task_result(data, "other1350", "running")
    assert client.get(_url("other1350", name, ref["path"])).status_code == 404, "another task's store is not this one's"
    assert client.get(_url("missing1350", name, ref["path"])).status_code == 404
    (folder / name).write_bytes(b"tampered\n")
    tampered = client.get(_url(TASK, name, ref["path"]))
    assert tampered.status_code == 404 and tampered.json()["reason_code"] == "artifact_unverified"


@confined
def test_a_source_only_an_own_drive_holds_is_served_from_it_and_honestly_gone_after_it(tmp_path):
    data = tmp_path / "data"
    drive = data / "task_drives" / TASK
    drive.mkdir(parents=True)
    write_task_result(data, TASK, "running", child_drive_root=str(drive))
    ref = artifacts.store_actor_source_bytes(drive, TASK, category=delegate_activity.SOURCE_CATEGORY,
                                             source_id="run-1-1-1", data=b'{"seq": 1}\n', extension="jsonl")
    name = ref["path"].rsplit("/", 1)[1]
    client = _client(data)
    assert client.get(_url(TASK, name, ref["path"])).content == b'{"seq": 1}\n'
    shutil.rmtree(drive)
    gone = client.get(_url(TASK, name, ref["path"]))
    assert gone.status_code == 404, "no copy is invented once the only store holding it is pruned"


@pytest.mark.parametrize("confined_opens", [False, True])
@pytest.mark.parametrize("carrier", ["completion_observations", "acceptance_debt"])
def test_a_result_published_source_keeps_its_existing_contract(tmp_path, monkeypatch, confined_opens, carrier):
    monkeypatch.setattr(task_archive, "CONFINED", confined_opens)
    data = tmp_path / "data"
    data.mkdir()
    ref = artifacts.store_actor_source_bytes(data, TASK, category="context_checkpoints", source_id="panel",
                                             data=b'{"ok": true}', extension="json")
    write_task_result(data, TASK, "completed", **{carrier: {"source_ref": ref}})
    client = _client(data)
    name = ref["path"].rsplit("/", 1)[1]
    published = client.get(_url(TASK, name, ref["path"]))
    assert published.status_code == 200 and published.json() == {"ok": True}
    assert published.headers["content-type"] == "application/json"
    other = artifacts.store_actor_source_bytes(data, TASK, category="tool_results", source_id="private",
                                               data=b"private", extension="txt")
    other_name = other["path"].rsplit("/", 1)[1]
    assert client.get(_url(TASK, other_name, other["path"])).status_code == 404, "an unpublished source stays private"


def test_without_confined_opens_the_new_source_refuses_without_a_fallback_read(tmp_path, monkeypatch):
    data, _drive, _daemon, record, long_text = _retained(tmp_path)
    ref = record["source"]["ref"]
    name = ref["path"].rsplit("/", 1)[1]
    monkeypatch.setattr(task_archive, "CONFINED", False)   # the #1297 platforms
    client = _client(data)
    def no_read(*_args, **_kwargs):
        pytest.fail("unsupported confinement must not fall back to path-based or descriptor reads")

    monkeypatch.setattr(artifacts, "read_actor_source_bytes", no_read)
    monkeypatch.setattr(task_archive, "_open_member", no_read)

    def unavailable(source_name, source_path):
        response = client.get(_url(TASK, source_name, source_path))
        assert response.status_code == 503 and response.json()["reason_code"] == "artifact_unavailable"
        assert "x-ouroboros-artifact-identity" not in response.headers
        assert "x-ouroboros-artifact-sha256" not in response.headers
        assert long_text not in response.text and "tampered" not in response.text

    unavailable(name, ref["path"])
    folder = data / "task_results" / "artifacts" / TASK / "source_handles" / "delegated_activity"
    forged_name = name.replace(ref["sha256"], "0" * 64)
    shutil.copyfile(folder / name, folder / forged_name)
    unavailable(forged_name, f"source_handles/delegated_activity/{forged_name}")
    (folder / name).write_bytes(b"tampered\n")
    unavailable(name, ref["path"])
    assert client.get(_url(TASK, "other.jsonl", ref["path"])).status_code == 400
    for source in (f"source_handles/delegated_activity/../delegated_activity/{name}",
                   f"source_handles/tool_results/{name}", f"/{ref['path']}"):
        assert client.get(_url(TASK, name, source)).status_code == 404
    assert client.get(_url("missing1350", name, ref["path"])).status_code == 404
