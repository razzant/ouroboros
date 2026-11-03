"""#1297 supported writer authority observations, not an OS confinement proof."""
from __future__ import annotations

import pytest
import json
import sys
from pathlib import Path

from ouroboros import artifacts, headless
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tools.core import _write_file
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.registry import ToolRegistry
from ouroboros.task_results import write_task_result
from tests.test_task_download_confinement import _http, _url, TASK
from tests import test_tool_access_matrix_closure as _closure

world = _closure.world  # the closure suite's isolated fixture object, registered here by name


def test_artifact_write_binding_cannot_select_child_store_or_ancestors(world):
    child = headless.prepare_task_drive(world.data, "t1", "empty")
    target = child / "state/current.sqlite"
    target.write_bytes(b"synthetic store marker; not a database")
    ctx = ToolContext(repo_dir=world.repo, drive_root=child, task_id="t1",
                      task_metadata={"budget_drive_root": str(world.data)})
    assert _write_file(ctx, root="artifact_store", path="report.txt", content="ordinary output").startswith("OK:")
    for path in ("../../../state/current.sqlite", str(target), str(child), str(child.parent)):
        result = _write_file(ctx, root="artifact_store", path=path, content="ATTEMPTED STORE WRITE")
        assert not result.startswith("OK:"), (path, result)
    result = _write_file(ctx, root="artifact_store", path=".artifact_manifest.json", content="{}")
    assert "task metadata paths are reserved" in result
    assert target.read_bytes() == b"synthetic store marker; not a database"
    assert (artifacts.task_artifact_dir_path(child, "t1") / "report.txt").read_text() == "ordinary output"


@pytest.mark.parametrize("root", ["runtime_data", "task_drive", "artifact_store", "user_files"])
def test_acting_subagent_file_tools_have_no_generated_data_write_binding(world, root):
    ctx = ToolContext(repo_dir=world.repo, drive_root=world.data, task_id="t1",
                      task_constraint=TaskConstraint(mode="acting_subagent", surface="external_workspace",
                                                     write_root=str(world.repo)))
    result = _write_file(ctx, root=root, path="state/current.sqlite", content="ATTEMPTED STORE WRITE")
    assert not result.startswith("OK:"), result
    assert "ACCESS_BLOCKED" in result, result


@pytest.mark.parametrize("replay_old_discovery", [False, True])
def test_supported_task_scratch_is_not_a_sqlite_store(world, monkeypatch, record_property, replay_old_discovery):
    """Actual run_command writes only allowed scratch; no forged canonical root.

    True explicitly replays the old existence-only discovery predicate so this
    control proves the same input reaches SQLite without the admission filter.
    It is a controlled replay, not a separate baseline executable/native proof.
    """
    from ouroboros import store as current_store, task_custody
    from ouroboros.gateway import task_archive

    project = world.repo.parent / "external-project"
    project.mkdir()
    ctx = ToolContext(repo_dir=world.repo, drive_root=world.data, task_id=TASK,
                      workspace_root=project, workspace_mode="external")
    registry = ToolRegistry(repo_dir=world.repo, drive_root=world.data)
    registry.set_context(ctx)
    store = artifacts.task_artifact_dir_path(world.data, TASK, create=True)
    (store / "ready.txt").write_bytes(b"canonical bytes")
    write_task_result(world.data, TASK, "completed", artifacts=artifacts.collect_task_artifact_records(world.data, TASK))
    # A valid SQLite header suffices: any schema is untrusted, and the read must
    # not get as far as the SQLite connection. All files are relative to scratch.
    code = ("from pathlib import Path\nimport sqlite3\nPath('state').mkdir(exist_ok=True)\n"
            "c=sqlite3.connect('state/current.sqlite')\n"
            f"c.execute('PRAGMA application_id={current_store._APPLICATION_ID[current_store.TIER_ENFORCED]}')\n"
            "c.execute('CREATE TABLE planted(value TEXT)')\nc.commit()\nc.close()\n")
    made = registry.execute("run_command", {"cmd": [sys.executable, "-c", code], "cwd": "task_drive"})
    assert "exit_code=0" in made, made
    scratch = world.data / "task_drives" / TASK
    assert (scratch / "state/current.sqlite").is_file()
    attempts, connect = [], current_store.connect
    def observed(path, *args, **kwargs):
        if Path(path).is_relative_to(scratch):
            attempts.append(str(path))
        return connect(path, *args, **kwargs)
    monkeypatch.setattr(current_store, "connect", observed)
    if replay_old_discovery:
        original = task_custody.own_child_drives
        monkeypatch.setattr(task_custody, "own_child_drives", lambda root, tid, **kwargs: original(root, tid))
    with _http(world.data) as client:
        response = client.get(_url("ready.txt"))
    assert response.status_code == (200 if task_archive.CONFINED else 503)
    record_property("supported_scratch_sqlite", json.dumps({"controlled_old_discovery": replay_old_discovery,
                     "connections": attempts, "producer": "run_command cwd=task_drive; relative state/current.sqlite"}))
    assert bool(attempts) == replay_old_discovery, attempts


@pytest.mark.parametrize("layout", ["headless", "legacy"])
@pytest.mark.parametrize("status", ["completed", "cancelled"])
@pytest.mark.parametrize("root_field", ["child_drive_root", "headless_child_drive_root", "metadata"])
def test_admitted_child_claims_survive_cancellation_and_legacy_layout(world, layout, status, root_field):
    from ouroboros.gateway import task_archive
    data = world.data
    child = (headless.prepare_task_drive(data, TASK, "empty") if layout == "headless"
             else data / "task_drives" / TASK)
    directory = artifacts.task_artifact_dir_path(child, TASK, create=True)
    (directory / "claim.txt").write_bytes(b"captured")
    (directory / "errored.txt").write_bytes(b"must not serve")
    rows = artifacts.collect_task_artifact_records(child, TASK)
    next(row for row in rows if row["name"] == "errored.txt")["errors"] = ["copy failed"]
    write_task_result(child, TASK, "completed", artifacts=rows)
    # Admission/copy-back/legacy metadata spellings, with no canonical artifacts.
    field = {root_field: {"drive_root": str(child)} if root_field == "metadata" else str(child)}
    write_task_result(data, TASK, status, **field)
    with _http(data) as client:
        response = client.get(_url("claim.txt"))
        if not task_archive.CONFINED:
            assert response.status_code == 200, "native positive must expose the retained Windows gate"
        assert (response.status_code, response.content) == (200, b"captured")
        assert response.headers["x-ouroboros-artifact-identity"] == "verified"
        detail = client.get(f"/api/tasks/{TASK}").json()
        assert next(row for row in detail["artifacts"] if row["name"] == "errored.txt")["errors"] == ["copy failed"]
        (directory / "claim.txt").write_bytes(b"changed!")
        refused = client.get(_url("claim.txt"))
        assert (refused.status_code, refused.json()["reason_code"]) == (409, "artifact_identity_changed")


def test_store_discovery_checks_canonical_proof_before_directory_io(world, monkeypatch):
    from ouroboros import confined_files, task_custody
    (world.data / "task_drives" / TASK).mkdir()
    def unexpected(*args, **kwargs):
        pytest.fail("unadmitted directory reached metadata discovery")
    monkeypatch.setattr(confined_files, "confined_directory", unexpected)
    assert task_custody.own_child_drives(world.data, TASK, store_row={"task_id": TASK}) == []


@pytest.mark.parametrize("layout", ["headless", "legacy"])
def test_retry_occupant_store_requires_canonical_execution_fact(world, monkeypatch, layout):
    from ouroboros import source_retention, task_store
    child = (headless.prepare_task_drive(world.data, TASK, "empty") if layout == "headless"
             else world.data / "task_drives" / TASK)
    successor = "retry-successor"
    write_task_result(child, successor, "completed", result="retained retry")
    write_task_result(world.data, successor, "completed", original_task_id=TASK, timeout_retry_from=TASK)
    attempts, read = [], task_store.read_document
    def observed(root, *args, **kwargs):
        if Path(root) == child:
            attempts.append(str(root))
        return read(root, *args, **kwargs)
    monkeypatch.setattr(task_store, "read_document", observed)
    assert not source_retention.retained_child_root(world.data, child, successor)
    assert attempts == []
    write_task_result(world.data, successor, "completed", drive_root=str(child))
    assert source_retention.retained_child_root(world.data, child, successor)
    assert attempts == [str(child)]


@pytest.mark.parametrize("loose", [False, True])
@pytest.mark.parametrize("ordinary_gate_closed", [False, True], ids=["platform-gate", "closed-gate-portable-logic"])
def test_supported_scratch_source_never_opens_placement_index(
        world, monkeypatch, record_property, loose, ordinary_gate_closed):
    """Published scratch sources remain readable without granting SQLite authority.

    Forcing the ordinary file gate closed tests platform-independent branch logic,
    not native Windows behavior. Published source refs intentionally bypass it.
    """
    from ouroboros import evidence_packs as packs, store as current_store
    from ouroboros.gateway import task_archive
    from tests.test_task_source_confined_qa import _packed

    if ordinary_gate_closed:
        monkeypatch.setattr(task_archive, "CONFINED", False)
    ref, raw, _loose, pack, _ = _packed(world.repo.parent / "producer", TASK)
    project = world.repo.parent / "external-project"
    project.mkdir()
    registry = ToolRegistry(repo_dir=world.repo, drive_root=world.data)
    registry.set_context(ToolContext(repo_dir=world.repo, drive_root=world.data, task_id=TASK,
                                    workspace_root=project, workspace_mode="external"))
    files = {"observability/packs/" + pack.name: pack.read_bytes().hex(),
             "observability/packs/" + packs.sidecar_name(TASK, 1):
                 pack.with_name(packs.sidecar_name(TASK, 1)).read_bytes().hex()}
    if loose:
        files[f"task_results/artifacts/{TASK}/" + ref["path"]] = raw.hex()
    code = ("from pathlib import Path\nimport sqlite3\n"
            f"files={files!r}\n"
            "for name, raw in files.items():\n p=Path(name)\n p.parent.mkdir(parents=True,exist_ok=True)\n p.write_bytes(bytes.fromhex(raw))\n"
            "c=sqlite3.connect('observability/packs/index.sqlite')\n"
            "c.execute('CREATE TABLE planted(value TEXT)')\nc.commit()\nc.close()\n")
    made = registry.execute("run_command", {"cmd": [sys.executable, "-c", code], "cwd": "task_drive"})
    assert "exit_code=0" in made, made
    scratch = world.data / "task_drives" / TASK
    index = packs.packs_dir(scratch) / packs.INDEX_NAME
    original_index = index.read_bytes()
    write_task_result(world.data, TASK, "completed", completion_observations={"source_ref": ref})
    attempts, connect = [], current_store.connect
    def observed(path, *args, **kwargs):
        if Path(path).is_relative_to(scratch):
            attempts.append(str(path))
        return connect(path, *args, **kwargs)
    monkeypatch.setattr(current_store, "connect", observed)
    assert packs.handle_info(scratch, TASK, ref["path"], anchor=world.data) == {
        "size": len(raw), "sha256": ref["sha256"], "loose": loose}
    assert artifacts.read_actor_source_bytes(world.data, TASK, ref, anchor=world.data) == raw
    with _http(world.data) as client:
        response = client.get(_url(Path(ref["path"]).name), params={"source": ref["path"]})
    assert (response.status_code, response.content) == (200, raw)
    record_property("supported_scratch_source", json.dumps({"loose": loose, "sqlite_connections": attempts,
                    "ordinary_file_gate": task_archive.CONFINED,
                    "forced_closed_portable_logic": ordinary_gate_closed,
                    "producer": "run_command cwd=task_drive; relative packs, sidecar and index.sqlite"}))
    assert attempts == []
    assert index.read_bytes() == original_index
