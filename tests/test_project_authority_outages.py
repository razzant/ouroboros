"""Temporary authority faults keep the same accepted row waiting, never replayed or failed."""
from __future__ import annotations

import copy
import errno
import json
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from ouroboros import projects_registry as registry
from ouroboros.task_results import load_task_result, write_task_result
from supervisor import queue, workers
from tests.test_hurry_initial_lifecycle import pool  # noqa: F401
from tests.test_main_project_consistency import owner_turn
from tests.test_project_hold_recovery import accepted, restore_unreadable, resume_after_app_stop, worker
from tests.test_swarm_host_admission import host  # noqa: F401
from tests._store_shared import delete_result, result_exists, result_signature, seed_result, unreadable_result
from tests._cases import chosen_rows

def test_exact_registry_loss_preserves_every_row_and_binding(tmp_path, monkeypatch):
    from ouroboros import project_registry_recovery as recovery
    from ouroboros.project_admission import project_scope_admission
    from supervisor import state

    state.init(tmp_path)
    project = registry.create_project(tmp_path, "target", working_dir="/synthetic/folder")
    registry.update_project(tmp_path, "target", working_dir="/synthetic/new-folder")
    basis = registry.project_admission_view(tmp_path, "target", frozen=True)
    for pid in ("gone", "deleting"):
        registry.create_project(tmp_path, pid)
        registry.begin_project_deletion(tmp_path, pid)
    registry.complete_project_deletion(tmp_path, "gone")
    registry.create_project(tmp_path, "folderless")
    registry.bind_task_to_project(tmp_path, "task", "target", origin={"absent": "system"})
    path = registry._registry_path(tmp_path)
    original = path.read_bytes()
    bindings = registry._bindings_path(tmp_path).read_bytes()
    assert recovery.recovery_notice(tmp_path) is None
    path.unlink()
    for write in (lambda: registry.create_project(tmp_path, "other"),
                  lambda: registry.update_project(tmp_path, "target", name="rename"),
                  lambda: registry.bind_task_to_project(tmp_path, "later", "target", origin={"absent": "system"}),
                  lambda: registry.begin_project_deletion(tmp_path, "target"),
                  lambda: project_scope_admission(tmp_path, project_id="target")):
        with pytest.raises(FileNotFoundError):
            write()
    direct, receipts = owner_turn(tmp_path, monkeypatch, project["chat_id"], "web")
    assert not direct and receipts[-1]["status"] == "project_unavailable"
    assert not path.exists() and recovery.recovery_notice(tmp_path) is None
    registry.reconcile_projects(tmp_path)
    assert path.read_bytes() == original and registry._bindings_path(tmp_path).read_bytes() == bindings
    assert recovery.recovery_notice(tmp_path) == {"kind": "exact_registry_restored"}
    with registry.project_admission_guard(tmp_path, basis) as current:
        assert current["working_dir"] == "/synthetic/new-folder"
    registry.reconcile_projects(tmp_path)
    assert path.read_bytes() == original
    path.unlink()
    assert recovery.recovery_notice(tmp_path) is None


@pytest.mark.parametrize("proof", ["clean", "old_writer", "before_new_primary", "after_new_primary", "foreign",
                                   "bad_copy", "legacy", "name_tier", "body_rollback"])
def test_registry_loss_from_the_same_active_room_requires_proven_freshness(tmp_path, monkeypatch, proof):
    from ouroboros import platform_layer as platform, project_registry_recovery as recovery, utils

    if proof == "name_tier":
        monkeypatch.setattr(platform, "kernel_file_locks_enforced", lambda *args, **kwargs: False)
    if proof == "body_rollback":
        body = tmp_path / "project_registry_recovery.py"
        body.write_bytes(b"# synthetic upgraded body")
        monkeypatch.setattr(recovery, "__file__", str(body))
    active = registry.create_project(tmp_path, "target")
    path = registry._registry_path(tmp_path)
    witness = registry._registry_witness_path(tmp_path)
    before = path.read_bytes()
    if proof == "old_writer":
        # The base version takes this exact O_EXCL lock; it must enter immediately.
        # Crash after replacing the primary, BEFORE touching its witness. Its stale
        # lock is reclaimed at the next boot, after the usual grace has elapsed.
        lock = path.with_suffix(".json.lock")
        fd = platform.acquire_exclusive_file_lock(lock, timeout_sec=0)
        assert fd is not None
        try:
            utils.atomic_write_json(path, {"projects": [{**active, "lifecycle": "tombstoned"}]})
        finally:
            os.close(fd)
        os.utime(lock, ns=(0, 0))
    elif proof == "before_new_primary":
        with monkeypatch.context() as fault:
            def interrupted(target, *args, **kwargs):
                assert target == path and recovery.recovery_notice(tmp_path) is None
                raise OSError("Cut before primary replacement")
            fault.setattr(utils, "write_bytes_atomic", interrupted)
            with pytest.raises(OSError, match="Cut before"):
                registry.begin_project_deletion(tmp_path, "target")
        assert path.read_bytes() == before
    elif proof == "after_new_primary":
        with monkeypatch.context() as fault:
            def interrupted(*args):
                raise OSError("Cut after primary replacement")
            fault.setattr(recovery, "_qualify", interrupted)
            registry.begin_project_deletion(tmp_path, "target")
        assert json.loads(path.read_bytes())["projects"][0]["lifecycle"] == "deleting"
    elif proof == "foreign":
        import shutil
        other = tmp_path / "other-install"
        registry.create_project(other, "foreign")
        for item in (witness, path.with_suffix(".json.lock")):
            shutil.copyfile(other / "state" / item.name, item)
    elif proof == "bad_copy":
        witness.write_bytes(b'{"projects":[]}')
    elif proof == "legacy":
        witness.write_bytes(b"")
        path.with_suffix(".json.lock").unlink(missing_ok=True)
    elif proof == "body_rollback":
        # An older checkout lacks this module. Model even a pre-lock writer
        # retaining both the new witness and the new lock verbatim.
        body.rename(body.with_suffix(".previous"))
        utils.atomic_write_json(path, {"projects": [{**active, "lifecycle": "tombstoned"}]})
        body.write_bytes(b"# synthetic upgraded body")
    saved_witness = witness.read_bytes()
    path.unlink()
    store = tmp_path / "projects/target"
    store.mkdir(parents=True)
    marker = store / ".project.json"
    marker.write_text(json.dumps(active))
    registry._save_bindings(tmp_path, {"bindings": {"t": {"project_id": "target"}}})
    if proof == "clean":
        import subprocess
        import sys
        # A cold interpreter takes the real boot reconciliation step. No server,
        # worker or queue has been started in that process.
        subprocess.run([sys.executable, "-c",
                        "from pathlib import Path; import sys; from ouroboros.server_boot import reconcile_projects; "
                        "reconcile_projects(Path(sys.argv[1]))", str(tmp_path)], check=True)
        assert path.read_bytes() == before
        assert recovery.recovery_notice(tmp_path) == {"kind": "exact_registry_restored"}
        return
    registry.reconcile_projects(tmp_path)
    assert not path.exists() and witness.read_bytes() == saved_witness and marker.is_file()
    assert recovery.recovery_notice(tmp_path) is None
    with pytest.raises(FileNotFoundError):
        registry.create_project(tmp_path, "replacement")


def test_healthy_legacy_seed_is_byte_exact_and_rejects_malformed_sources(tmp_path):
    from ouroboros import project_registry_recovery as recovery

    path, witness = registry._registry_path(tmp_path), registry._registry_witness_path(tmp_path)
    path.parent.mkdir(parents=True)
    witness.touch()
    for broken in (b'{"projects":[null]}', b'{"projects":[],"projects":[]}', b'{"_schema_version":88,"projects":[]}'):
        path.write_bytes(broken)
        registry.reconcile_projects(tmp_path)
        assert path.read_bytes() == broken and witness.read_bytes() == b""
    # No cognition migration is owed by this already-healthy legacy row.
    legacy = b'{ "projects" : [{"id":"target","cognition":{},"opaque":{"keep":1}}] }\n'
    path.write_bytes(legacy)
    path.chmod(0o600)
    registry.reconcile_projects(tmp_path)
    assert witness.read_bytes() == legacy and path.read_bytes() == legacy
    assert recovery.recovery_notice(tmp_path) is None
    path.unlink()
    registry.reconcile_projects(tmp_path)
    assert path.read_bytes() == legacy and recovery.recovery_notice(tmp_path)
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == stat.S_IMODE(witness.stat().st_mode) == 0o600


def test_an_independently_returned_primary_wins_and_revokes_the_outdated_copy(tmp_path, monkeypatch):
    from ouroboros import project_registry_recovery as recovery

    registry.create_project(tmp_path, "target")
    path = registry._registry_path(tmp_path)
    path.unlink()
    publish = recovery._publish_new_file
    returned = b'{ "projects": [] }'
    with monkeypatch.context() as fault:
        def return_primary(source, target):
            target.write_bytes(returned)
            return publish(source, target)
        fault.setattr(recovery, "_publish_new_file", return_primary)
        registry.reconcile_projects(tmp_path)
    assert path.read_bytes() == returned and recovery.recovery_notice(tmp_path) is None
    path.unlink()
    registry.reconcile_projects(tmp_path)
    assert not path.exists()


def test_a_primary_replaced_during_seeding_cannot_certify_the_old_bytes(tmp_path, monkeypatch):
    from ouroboros import project_registry_recovery as recovery, utils

    path, witness = registry._registry_path(tmp_path), registry._registry_witness_path(tmp_path)
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b'{"projects":[{"id":"old","cognition":{}}]}')
    witness.touch()
    replace = utils.replace_atomic
    with monkeypatch.context() as fault:
        def put_back_the_original(src, dst, **kwargs):
            result = replace(src, dst, **kwargs)
            if dst == witness:
                path.write_bytes(b'{"projects":[]}')
            return result
        fault.setattr(utils, "replace_atomic", put_back_the_original)
        registry.reconcile_projects(tmp_path)
    assert json.loads(path.read_bytes()) == {"projects": []}
    path.unlink()
    registry.reconcile_projects(tmp_path)
    assert not path.exists() and recovery.recovery_notice(tmp_path) is None


@pytest.mark.parametrize("peer_kernel", [True, False])
def test_current_and_legacy_writers_exclude_each_other(tmp_path, monkeypatch, peer_kernel):
    import hashlib
    from ouroboros import platform_layer as platform, project_registry_recovery as recovery

    registry.create_project(tmp_path, "active")
    path = registry._registry_path(tmp_path)
    lock = path.with_suffix(".json.lock")
    magic, payload = lock.read_bytes().split(b"\n", 1)
    assert magic == b"ouroboros-project-registry-v1"
    assert json.loads(payload)["digest"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with recovery.writer_lock(path):
        with monkeypatch.context() as peer:
            peer.setattr(platform, "kernel_file_locks_enforced", lambda *_: peer_kernel)
            assert platform.acquire_exclusive_file_lock(lock, timeout_sec=0) is None
    with monkeypatch.context() as peer:
        peer.setattr(platform, "kernel_file_locks_enforced", lambda *_: peer_kernel)
        fd = platform.acquire_exclusive_file_lock(lock, timeout_sec=0)
    assert fd is not None
    try:
        with pytest.raises(TimeoutError), recovery.writer_lock(path, timeout=0):
            pytest.fail("recovery acquired an occupied legacy lock")
    finally:
        platform.release_exclusive_file_lock(lock, fd)


def test_code_update_renews_the_unchanged_primary_and_keeps_legacy_maintenance(tmp_path, monkeypatch):
    from ouroboros import project_registry_recovery as recovery

    body = tmp_path / "body.py"
    body.write_text("# earlier body")
    monkeypatch.setattr(recovery, "__file__", str(body))
    room = registry.create_project(tmp_path, "existing")
    (tmp_path / "projects" / "legacy").mkdir(parents=True)
    updated = tmp_path / "new.py"
    updated.write_text("# updated body")
    updated.replace(body)

    assert registry.reconcile_projects(tmp_path) == 1
    assert registry.get_project(tmp_path, "existing")["routing_incarnation"] == room["routing_incarnation"]
    assert recovery.recovery_notice(tmp_path) is None
    path = recovery.registry_path(tmp_path)
    current = path.read_bytes()
    path.unlink()
    registry.reconcile_projects(tmp_path)
    assert path.read_bytes() == current
    assert recovery.recovery_notice(tmp_path) == {"kind": "exact_registry_restored"}


def test_ordinary_maintenance_keeps_healthy_neighbours_of_a_partial_registry(tmp_path):
    path = registry._registry_path(tmp_path)
    path.parent.mkdir()
    invalid = {"id": "partial", "lifecycle": None, "history": "retained"}
    path.write_text(json.dumps({"projects": [
        invalid, {"id": "healthy"}, {"id": "gone", "lifecycle": "tombstoned"},
    ]}))
    retained = tmp_path / "projects/gone/.project.json"
    retained.parent.mkdir(parents=True)
    retained.write_text('{"id":"gone"}')
    (tmp_path / "projects/other").mkdir()
    assert registry.reconcile_projects(tmp_path) == 0
    rows = json.loads(path.read_bytes())["projects"]
    assert rows[0] == invalid
    assert rows[1]["cognition"] and rows[2]["lifecycle"] == "tombstoned"
    assert not retained.exists()
    assert {row["id"] for row in rows} == {"healthy", "gone", "partial"}
    assert registry._registry_witness_path(tmp_path).read_bytes() == b""


def test_unchanged_registry_tick_reads_one_snapshot_and_keeps_its_witness(tmp_path, monkeypatch):
    from ouroboros import project_registry_recovery as recovery, utils

    room = registry.create_project(tmp_path, "existing")
    path, witness = recovery.registry_path(tmp_path), recovery.witness_path(tmp_path)
    original = path.read_bytes()
    witnessed = utils.file_identity(witness.stat())
    decode = recovery.read_bytes
    decoded = []

    def count_snapshot(raw):
        decoded.append(raw)
        return decode(raw)

    monkeypatch.setattr(recovery, "read_bytes", count_snapshot)
    assert registry.reconcile_projects(tmp_path) == 0
    assert decoded == [original]
    assert utils.file_identity(witness.stat()) == witnessed
    assert registry.get_project(tmp_path, "existing")["routing_incarnation"] == room["routing_incarnation"]


@pytest.mark.skipif(os.name == "nt", reason="Windows rename publishes only to an absent name")
def test_linkless_volume_publishes_only_to_an_absent_name(tmp_path, monkeypatch):
    from ouroboros import project_registry_recovery as recovery

    def linkless(*args):
        raise OSError(errno.ENOTSUP, "no hard links on this volume")

    monkeypatch.setattr(os, "link", linkless)
    room = registry.create_project(tmp_path, "first")
    assert registry.get_project(tmp_path, "first")["chat_id"] == room["chat_id"]
    path = recovery.registry_path(tmp_path)
    original = path.read_bytes()
    path.unlink()
    registry.reconcile_projects(tmp_path)
    assert path.read_bytes() == original and recovery.recovery_notice(tmp_path)
    assert registry.get_project(tmp_path, "first")["routing_incarnation"] == room["routing_incarnation"]
    staged = path.with_name("synthetic-staged")
    staged.write_bytes(b"other")
    with pytest.raises(FileExistsError):
        recovery._publish_new_file(staged, path)
    assert path.read_bytes() == original and staged.read_bytes() == b"other"


def test_registry_write_guard_precedes_creation_of_state_or_locks(tmp_path, monkeypatch):
    from ouroboros import project_registry_recovery as recovery

    root = tmp_path / "guarded"
    monkeypatch.setenv("OUROBOROS_PYTEST_ACTIVE", "1")
    monkeypatch.delenv("OUROBOROS_ALLOW_LIVE_DATA_TESTS", raising=False)
    monkeypatch.setenv("OUROBOROS_TEST_LIVE_DATA_ROOT", str(root))
    with pytest.raises(RuntimeError, match="PYTEST_LIVE_DATA_WRITE_BLOCKED"):
        recovery.initialize_new_root(root)
    assert not root.exists()


def test_first_lock_probe_cannot_certify_a_name_only_writer(tmp_path, monkeypatch):
    from ouroboros import platform_layer as platform, project_registry_recovery as recovery

    path = registry._registry_path(tmp_path)
    registry_lock = path.with_suffix(".json.lock")
    normal_tier = platform.kernel_file_locks_enforced
    probes = iter((True, False))

    def lock_tier(lock, **kwargs):
        return next(probes, False) if lock == registry_lock else normal_tier(lock, **kwargs)

    monkeypatch.setattr(platform, "kernel_file_locks_enforced", lock_tier)
    registry.create_project(tmp_path, "target")
    witness = registry._registry_witness_path(tmp_path)
    assert witness.read_bytes() == b""
    path.unlink()
    registry.reconcile_projects(tmp_path)
    assert not path.exists() and recovery.recovery_notice(tmp_path) is None


def test_corruption_after_an_exact_restore_preserves_the_primary_and_hides_the_notice(tmp_path):
    from ouroboros import project_registry_recovery as recovery

    registry.create_project(tmp_path, "target")
    path = registry._registry_path(tmp_path)
    path.unlink()
    registry.reconcile_projects(tmp_path)
    assert recovery.recovery_notice(tmp_path)
    path.write_bytes(b"{")
    assert recovery.recovery_notice(tmp_path) is None
    registry.reconcile_projects(tmp_path)
    assert path.read_bytes() == b"{" and recovery.recovery_notice(tmp_path) is None


@pytest.mark.parametrize("receipt", ["recorded", "interrupted"])
def test_legacy_store_waits_through_repeated_ticks_of_a_restored_revision(tmp_path, monkeypatch, receipt):
    from ouroboros import project_registry_recovery as recovery, utils

    registry.create_project(tmp_path, "kept")
    path = registry._registry_path(tmp_path)
    saved = path.read_bytes()
    (tmp_path / "projects" / "legacy-store" / "knowledge").mkdir(parents=True)
    path.unlink()
    with monkeypatch.context() as fault:
        if receipt == "interrupted":
            write = utils.atomic_write_json

            def cut_receipt(target, data, **kwargs):
                if target == recovery._restore_receipt(path) and "primary" in data:
                    raise OSError("notice write interrupted")
                return write(target, data, **kwargs)

            fault.setattr(utils, "atomic_write_json", cut_receipt)
        assert registry.reconcile_projects(tmp_path) == 0
    restored = utils.file_identity(path.stat())
    for _ in range(2):
        assert registry.reconcile_projects(tmp_path) == 0
        assert path.read_bytes() == saved and utils.file_identity(path.stat()) == restored
        assert bool(recovery.recovery_notice(tmp_path)) == (receipt == "recorded")
    registry.update_project(tmp_path, "kept", name="Renamed")
    assert recovery.recovery_notice(tmp_path) is None
    assert registry.reconcile_projects(tmp_path) == 1
    assert {row["id"] for row in registry.list_projects(tmp_path)} == {"kept", "legacy-store"}


def test_malformed_primary_cannot_import_a_named_legacy_store(tmp_path):
    path = registry._registry_path(tmp_path)
    path.parent.mkdir()
    invalid = b'{"projects":[{"id":"held","lifecycle":null}]}'
    path.write_bytes(invalid)
    store = tmp_path / "projects" / "legacy-store" / "knowledge"
    store.mkdir(parents=True)
    assert registry.reconcile_projects(tmp_path) == 0
    assert path.read_bytes() == invalid and store.is_dir()


def test_first_explicit_creation_requires_a_root_without_project_evidence(tmp_path):
    assert not registry._registry_path(tmp_path).exists()
    assert registry.reconcile_projects(tmp_path) == 0
    assert registry.create_project(tmp_path, "fresh")["created"] is True
    legacy = tmp_path / "legacy"
    (legacy / "projects" / "old").mkdir(parents=True)
    registry.reconcile_projects(legacy)
    with pytest.raises(FileNotFoundError):
        registry.create_project(legacy, "new")


def test_never_committed_install_still_creates_and_reconciles_rooms(tmp_path):
    from supervisor.state_initialization import initialization_decision

    assert initialization_decision(tmp_path)["create"]
    (tmp_path / "projects" / "legacy").mkdir(parents=True)
    assert registry.reconcile_projects(tmp_path) == 1
    assert registry._registry_witness_path(tmp_path).exists()
    assert registry.create_project(tmp_path, "fresh")["created"] is True
    assert {row["id"] for row in registry.list_reserved_projects(tmp_path, strict=True)} == {"legacy", "fresh"}


@pytest.mark.parametrize("cut", ["before_first_primary", "after_first_primary", "before_first_room"])
def test_interrupted_first_registry_write_allows_retry(tmp_path, monkeypatch, cut):
    from ouroboros import project_registry_recovery as recovery, utils

    path, witness = registry._registry_path(tmp_path), registry._registry_witness_path(tmp_path)
    write, sync = utils._write_fd_fully, utils.fsync_folder

    def cut_write(fd, raw, target):
        if target.parent == path.parent and target.name.startswith(f".{path.name}.tmp."):
            rows = json.loads(raw)["projects"]
            if (not rows and cut == "before_first_primary") or (rows and cut == "before_first_room"):
                raise OSError("first write interrupted")
        return write(fd, raw, target)

    def cut_sync(folder):
        if cut == "after_first_primary" and folder == path.parent and path.is_file() and not witness.exists():
            raise OSError("first write interrupted")
        return sync(folder)

    with monkeypatch.context() as fault:
        fault.setattr(utils, "_write_fd_fully", cut_write)
        fault.setattr(utils, "fsync_folder", cut_sync)
        with pytest.raises(OSError, match="first write interrupted"):
            registry.create_project(tmp_path, "first")
    if cut == "before_first_primary":
        assert not path.exists()
    else:
        assert json.loads(path.read_bytes())["projects"] == []
    assert witness.exists() == (cut == "before_first_room")
    assert recovery.recovery_notice(tmp_path) is None
    assert registry.create_project(tmp_path, "first")["created"] is True
    assert [row["id"] for row in json.loads(path.read_bytes())["projects"]] == ["first"]


@pytest.mark.parametrize("history", [
    "first_boot", "first_boot_linkless", "empty_primary_lost", "shelf_before_boot", "completed_boot",
])
def test_first_project_with_a_bare_workspace_shelf(tmp_path, monkeypatch, history):
    from ouroboros import config, project_facts, project_registry_recovery as recovery
    from supervisor import state_initialization as initialization

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    path = registry._registry_path(tmp_path)
    first_boot = history in {"first_boot", "first_boot_linkless", "empty_primary_lost"}
    if history == "first_boot_linkless":
        def unsupported_link(*args):
            raise OSError(errno.ENOTSUP, "hard links unavailable")
        monkeypatch.setattr(os, "link", unsupported_link)
    if first_boot:
        assert initialization.initialization_decision(tmp_path)["create"]
        assert json.loads(path.read_bytes())["projects"] == []
    elif history == "completed_boot":
        identity = initialization.mark_pending(tmp_path, origin="first_boot")
        assert initialization.complete(tmp_path, identity, adopted=False)

    pid = project_facts.resolve_project_id({"workspace_root": str(tmp_path / "bare-workspace")})
    assert pid.startswith("proj_")
    shelf = project_facts.project_knowledge_dir(pid)
    shelf.mkdir(parents=True)
    note = shelf / "fact.md"
    note.write_bytes(b"independent workspace fact")
    if history == "empty_primary_lost":
        path.unlink()
    if not first_boot:
        decision = initialization.initialization_decision(tmp_path)
        assert decision["create"] == (history == "shelf_before_boot")
    registry.reconcile_projects(tmp_path)
    assert bool(recovery.recovery_notice(tmp_path)) == (history == "empty_primary_lost")
    assert registry.reserved_project_chat_ids(tmp_path, strict=True) == set()
    assert registry.create_project(tmp_path, "first")["created"] is True
    assert [row["id"] for row in json.loads(path.read_bytes())["projects"]] == ["first"]
    assert note.read_bytes() == b"independent workspace fact"
    assert not (shelf.parent / ".project.json").exists()


def test_empty_legacy_witness_alone_holds_through_first_boot(tmp_path):
    from ouroboros import project_registry_recovery as recovery
    from supervisor import state_initialization as initialization

    path, witness = registry._registry_path(tmp_path), registry._registry_witness_path(tmp_path)
    path.parent.mkdir()
    witness.touch()
    assert initialization.initialization_decision(tmp_path)["create"]
    registry.reconcile_projects(tmp_path)
    with pytest.raises(FileNotFoundError):
        registry.create_project(tmp_path, "first")
    assert not path.exists() and witness.read_bytes() == b""
    assert recovery.recovery_notice(tmp_path) is None


@pytest.mark.parametrize("veto", [None, "possible", "started"])
def test_first_restore_keeps_accepted_project_row_waiting_through_unreadable_result(host, tmp_path, monkeypatch, veto):  # noqa: F811
    from ouroboros.gateway.state import _chat_activities_snapshot_safe
    from ouroboros.project_admission import project_hold_fact

    prepared = copy.deepcopy(accepted(host, tmp_path))
    accepted(host, tmp_path, tid="sibling")
    assert prepared["admitted_dispatch"] == "none" and "_project_admission_restore_hold" not in prepared
    original = result_signature(host.root, "held")
    assert queue.persist_queue_snapshot()
    unreadable = unreadable_result(monkeypatch, "held")
    for _restart in range(2):  # the persisted hold survives a second restart during the fault
        host.pending.clear()
        assert queue.restore_pending_from_snapshot() == 2
        held = next(row for row in host.pending if row["id"] == "held")
        assert not held.get("_terminalization_retry")
        assert project_hold_fact(held)["label"] == "Waiting for Project verification"
        assert queue.persist_queue_snapshot()
    census = {row["activity_id"]: row for row in _chat_activities_snapshot_safe(host.root, availability={})}
    # The same accepted id stays visible; its unreadable result is also unreadable
    # Pause authority, so Batch4 reports `unknown` rather than a guessed queue phase.
    assert census["held"]["phase"] == "unknown"
    assert census["held"]["project_admission_hold"]["label"] == "Waiting for Project verification"
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert not sent
    resume_after_app_stop(host, "sibling")
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["sibling"]  # a healthy neighbor is not blocked
    queue.RUNNING.clear()
    workers.WORKERS[0].busy_task_id = None
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["sibling"] and [row["id"] for row in host.pending] == ["held"]
    unreadable["on"] = False
    assert result_signature(host.root, "held") == original  # the original receipt is readable again, unchanged
    if veto == "possible":
        write_task_result(host.root, "held", "scheduled", admitted_dispatch="possible")
    elif veto == "started":
        write_task_result(host.root, "held", "scheduled", started_at="2026-01-01T00:00:00Z")
    # Selecting the app-stop hold does not discharge Project dispatch evidence:
    # the original admission checks below must still prevent either replay.
    resume_after_app_stop(host, "held")
    workers.assign_tasks()
    workers.assign_tasks()
    assert [row["id"] for row in sent] == (["sibling"] if veto else ["sibling", "held"])
    assert load_task_result(host.root, "held")["status"] == ("scheduled" if veto else "running")  # never failed
    assert not host.attempts
    if veto:
        assert host.pending[0]["_project_admission_restore_hold"]
    else:
        assert sent[1]["_project_admission"] == prepared["_project_admission"]
        assert sent[1]["workspace_root"] == prepared["workspace_root"]
        assert sent[1]["drive_root"] == prepared["drive_root"]


def _uncertain_child(host, monkeypatch):  # noqa: F811
    """The real child admission whose committed receipt cannot be read back (`_admission_uncertain`)."""
    from supervisor import task_admission

    write_task_result(host.root, "parent", "running", chat_id=1)
    child = {"id": "held", "type": "task", "text": "Child work", "chat_id": 1, "project_id": "target",
             "root_task_id": "parent", "parent_task_id": "parent", "delegation_role": "subagent", "depth": 1,
             "_project_admission": registry.project_admission_view(host.root, "target", frozen=True)}
    committed, write = [], task_admission.write_task_result

    def write_then_tear(root, tid, *args, **kwargs):
        stored = write(root, tid, *args, **kwargs)
        committed.append(result_signature(root, tid))
        committed.append(unreadable_result(monkeypatch, tid))  # committed, then unreadable before its readback
        return stored

    monkeypatch.setattr(task_admission, "write_task_result", write_then_tear)
    admitted, reason, _detail, _persist = task_admission.enqueue_subagent_with_scheduled_result(
        host.ctx, child, result_fields={"chat_id": 1}, admitted_task_contract={}, admitted_depth_provenance={},
        direct_child_count=0, pending_ref=host.pending)
    monkeypatch.setattr(task_admission, "write_task_result", write)
    assert not reason and admitted["_admission_uncertain"] and admitted in host.pending
    return copy.deepcopy(admitted), committed[0], committed[1]


@pytest.mark.parametrize("producer", ["promotion", "child_uncertain"])
@pytest.mark.parametrize("veto", [None, "stop", "terminal"])
def test_live_accepted_project_row_waits_through_its_first_unreadable_result(host, tmp_path, monkeypatch,  # noqa: F811
                                                                              producer, veto):
    """R2: a normally accepted, never-restored Project row whose own result first becomes
    unreadable keeps its id, payload and resources under the existing hold instead of failed
    custody; the healthy neighbor proceeds. Once readable, the original receipt releases it
    exactly once, while Stop or a terminal result settles it without a handoff."""
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.gateway.state import _chat_activities_snapshot_safe

    accepted(host, tmp_path, tid="sibling")
    if producer == "promotion":
        prepared = copy.deepcopy(accepted(host, tmp_path))
        original = result_signature(host.root, "held")
        unreadable = unreadable_result(monkeypatch, "held")
    else:
        prepared, original, unreadable = _uncertain_child(host, monkeypatch)
    assert prepared["admitted_dispatch"] == "none" and "_project_admission_restore_hold" not in prepared
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["sibling"]  # a healthy neighbor is not blocked
    held = next(row for row in host.pending if row["id"] == "held")
    assert held["_project_admission_restore_hold"] and not held.get("_terminalization_retry")
    if producer == "promotion":  # the census lists roots; a child's wait shows on its parent's tree
        census = {row["activity_id"]: row for row in _chat_activities_snapshot_safe(host.root, availability={})}
        assert census["held"]["project_admission_hold"]["label"] == "Waiting for Project verification"
    snapshot = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert next(row for row in snapshot["pending"] if row["id"] == "held")["task"]["_project_admission_restore_hold"]
    queue.RUNNING.clear()
    workers.WORKERS[0].busy_task_id = None
    unreadable["on"] = False  # the original receipt is readable again
    assert result_signature(host.root, "held") == original
    if veto == "stop":
        request_cancel(host.root, "held", reason="owner stopped")
    elif veto == "terminal":
        write_task_result(host.root, "held", "failed", result="Settled by its own owner.")
    for _pass in range(3):
        workers.assign_tasks()
        queue.RUNNING.clear()
        workers.WORKERS[0].busy_task_id = None
    assert [row["id"] for row in sent] == (["sibling"] if veto else ["sibling", "held"])  # never replayed
    assert "held" not in {row["id"] for row in host.pending} and not host.attempts
    stored = load_task_result(host.root, "held")
    # A dispatched child without a prepared drive keeps its receipt status (no RUNNING mirror).
    assert stored["status"] == ({"stop": "cancelled", "terminal": "failed"}.get(veto)
                                or ("running" if producer == "promotion" else "scheduled"))
    if not veto:
        for key in ("_project_admission", "workspace_root", "drive_root", "text"):
            assert sent[1].get(key) == prepared.get(key)


RESOLVER = "update_assisted_merge_test"


def _resolver(root):
    """The apply's fresh transaction: no resolver submitted yet, so its first admission is provable."""
    from supervisor import update_merge

    tx = {"task_id": RESOLVER, "phase": "assisted_resolution", "owner_chat_id": 1,
          "target_sha": "target", "pre_update_sha": "baseline", "resolver_submitted_id": ""}
    update_merge.write_update_tx(tx)
    assert update_merge.enqueue_assisted_resolution_task(tx) == RESOLVER
    return tx


def _drain(slot):
    sent = []
    while not slot.in_q.empty():
        sent.append(slot.in_q.get_nowait()["id"])
    return sent


def _acknowledged_resolver_restart(root, pending):
    """These update consumers return through the planned-restart door, not a crash."""
    from supervisor.restart_retention import prepare_restart_returns
    from tests.test_restart_saved_work import _ack

    prepare_restart_returns(root, {}, pending, transaction_id="resolver-restart")
    _ack(root, "resolver-restart")


@pytest.mark.parametrize("prior", [None, "possible"])
def test_assisted_resolver_recovers_same_id_after_restore_scope_hold(pool, prior):  # noqa: F811
    """Unreadable bindings hold the restored resolver; once scope is verifiable it runs once,
    under its first-dispatch receipt or, if it may have reached a worker, boot recovery's resume."""
    from supervisor import update_merge

    tx = _resolver(pool.root)
    [row] = queue.PENDING
    assert row["_project_scope_none"] is True and row["admitted_dispatch"] == "none"
    receipt = load_task_result(pool.root, RESOLVER, strict=True)
    assert receipt["status"] == "scheduled" and receipt["host_admission"]["status"] == "accepted"
    assert queue.persist_queue_snapshot()
    _acknowledged_resolver_restart(pool.root, queue.PENDING)
    if prior == "possible":  # the resolver may already have reached a worker
        write_task_result(pool.root, RESOLVER, "scheduled", admitted_dispatch="possible")
    bindings = registry._bindings_path(pool.root)
    bindings.parent.mkdir(parents=True, exist_ok=True)
    bindings.write_text("{torn", encoding="utf-8")
    queue.PENDING.clear()
    queue.restore_pending_from_snapshot()
    assert update_merge.enqueue_assisted_resolution_task(tx) == RESOLVER  # boot recovery refreshes it
    [held] = queue.PENDING
    assert held["id"] == RESOLVER and held["_project_admission_restore_hold"]
    workers.assign_tasks()
    assert not _drain(pool.slot)
    bindings.write_text('{"bindings": {}}', encoding="utf-8")
    workers.assign_tasks()
    workers.assign_tasks()
    assert _drain(pool.slot) == [RESOLVER] and not queue.PENDING
    stored = load_task_result(pool.root, RESOLVER, strict=True)
    assert stored["host_admission"] == receipt["host_admission"]  # no fresh receipt was minted
    assert queue.RUNNING[RESOLVER]["task"]["admitted_dispatch"] == "possible"
    assert "_managed_update_resume" not in queue.RUNNING[RESOLVER]["task"]


def test_resolver_boot_reenqueue_of_an_existing_id_fabricates_no_receipt(pool):  # noqa: F811
    write_task_result(pool.root, RESOLVER, "interrupted", result="mid-flight restart",
                      admitted_dispatch="possible")
    before = result_signature(pool.root, RESOLVER)
    _resolver(pool.root)
    assert [row["id"] for row in queue.PENDING] == [RESOLVER]
    assert queue.PENDING[0]["admitted_dispatch"] == "possible"  # never a first-dispatch claim
    assert result_signature(pool.root, RESOLVER) == before
    workers.assign_tasks()
    assert _drain(pool.slot) == [RESOLVER]  # the managed resume still runs its resolver


def _unadmitted(root):
    rows = [json.loads(line) for line in (root / "logs" / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()]
    return [row["reason"] for row in rows if row["type"] == "managed_update_assisted_resolver_unadmitted"]


def test_resolver_never_mints_a_receipt_over_an_unreadable_prior(pool, monkeypatch):  # noqa: F811
    """A prior row the store cannot answer for: the strict read refuses, nothing is minted over
    it. (A row naming another task or a newer schema cannot enter the store: the import judges
    such files, the store keys every row by its own id.)"""
    from supervisor import update_merge

    seed_result(pool.root, RESOLVER, {"task_id": RESOLVER, "status": "running"})
    original = result_signature(pool.root, RESOLVER)
    unreadable = unreadable_result(monkeypatch, RESOLVER)
    tx = {"task_id": RESOLVER, "phase": "assisted_resolution", "owner_chat_id": 1, "target_sha": "target"}
    update_merge.write_update_tx(tx)
    assert update_merge.enqueue_assisted_resolution_task(tx) == ""  # possibly sent before: no replay
    unreadable["on"] = False
    assert not queue.PENDING and result_signature(pool.root, RESOLVER) == original  # the unknown row keeps its custody
    assert tx["task_id"] == RESOLVER and _unadmitted(pool.root) == ["task_result_unreadable"]


@pytest.mark.parametrize("prior", [None, "interrupted"])
def test_resolver_refused_by_its_admission_is_never_claimed_started(pool, prior):  # noqa: F811
    from supervisor import update_merge

    if prior:
        write_task_result(pool.root, RESOLVER, prior, admitted_dispatch="possible")
    before = load_task_result(pool.root, RESOLVER)
    bindings = registry._bindings_path(pool.root)
    bindings.parent.mkdir(parents=True, exist_ok=True)
    bindings.write_text("{torn", encoding="utf-8")
    tx = {"task_id": RESOLVER, "phase": "assisted_resolution", "owner_chat_id": 1, "target_sha": "target",
          "resolver_submitted_id": ""}  # the apply's transaction: nothing submitted yet
    update_merge.write_update_tx(tx)
    assert update_merge.enqueue_assisted_resolution_task(tx) == ""
    assert not queue.PENDING and load_task_result(pool.root, RESOLVER) == before  # no receipt, no row
    assert _unadmitted(pool.root) == ["project_routing_fence_lookup_failed"]
    assert update_merge.read_update_tx()["resolver_submitted_id"] == ""  # a proven refusal submitted nothing
    bindings.write_text('{"bindings": {}}', encoding="utf-8")
    assert update_merge.enqueue_assisted_resolution_task(tx) == RESOLVER  # the next attempt admits it
    workers.assign_tasks()
    assert _drain(pool.slot) == [RESOLVER]


@pytest.mark.parametrize(("loss", "queued"), [(None, False), ("removed", True), ("legacy", True)])
def test_resolver_whose_dispatch_evidence_is_lost_is_never_replayed(pool, loss, queued):  # noqa: F811
    """A resolver that reached a worker and then lost its result may already have run:
    the managed resume neither sends it again nor mints a first-dispatch receipt, whether
    boot finds no row or a row restored from a snapshot older than that handoff. The
    control (no loss, restart before the handoff) resumes the same id exactly once."""
    from supervisor import update_merge

    _resolver(pool.root)
    assert update_merge.read_update_tx()["resolver_submitted_id"] == RESOLVER  # written ahead
    assert load_task_result(pool.root, RESOLVER)["status"] == "scheduled"  # its provable first
    older = queue.QUEUE_SNAPSHOT_PATH.read_bytes() if queue.persist_queue_snapshot() else b""
    if loss:
        workers.assign_tasks()
        assert _drain(pool.slot) == [RESOLVER]  # the first resolver reached a worker
        assert load_task_result(pool.root, RESOLVER)["admitted_dispatch"] == "possible"
        queue.RUNNING.clear()  # the restart took it down with its worker
        pool.slot.busy_task_id = None
        delete_result(pool.root, RESOLVER)  # the row is gone (a lost store, an older snapshot's world)
        if loss == "legacy":  # a transaction written before submissions were recorded
            update_merge.write_update_tx({key: value for key, value in update_merge.read_update_tx().items()
                                          if key != "resolver_submitted_id"})
        assert not result_exists(pool.root, RESOLVER)
    tx = update_merge.read_update_tx()
    queue.PENDING.clear()
    if queued:  # boot restores the row from the snapshot persisted before the handoff
        queue.QUEUE_SNAPSHOT_PATH.write_bytes(older)
        _acknowledged_resolver_restart(pool.root, [entry["task"] for entry in json.loads(older)["pending"]])
        assert queue.restore_pending_from_snapshot() == 1
    assert update_merge.enqueue_assisted_resolution_task(tx) == ("" if loss else RESOLVER)
    workers.assign_tasks()
    workers.assign_tasks()
    assert _drain(pool.slot) == ([] if loss else [RESOLVER])  # at most one handoff per id
    if not loss:
        return
    assert _unadmitted(pool.root) == ["task_result_missing"] and not result_exists(pool.root, RESOLVER)  # no invented receipt
    assert update_merge.read_update_tx().get("resolver_submitted_id") == (None if loss == "legacy" else RESOLVER)
    if queued:  # the same accepted row, its authority refreshed, waits visibly: never replayed
        [held] = queue.PENDING
        assert held["id"] == RESOLVER and held["_project_admission_restore_hold"]
        assert update_merge.assisted_task_metadata_authorizes(tx, held["metadata"])
    else:
        assert not queue.PENDING


def _boot_recovery(pool, tmp_path, monkeypatch):  # noqa: F811
    """A live materialized assisted merge whose boot recovery reaches the REAL resolver admission."""
    from supervisor import git_ops, update_merge
    from tests import test_update_merge_assisted as tua

    repo, _head, _plan, tx = tua._materialized_conflict_tx(tmp_path, monkeypatch)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", pool.root)  # one root for results, receipts and logs
    update_merge.write_update_tx({**tx, "task_id": RESOLVER, "owner_chat_id": 1, "resolver_submitted_id": ""})
    return repo


def _supervisor_types(root):
    return [json.loads(line)["type"] for line in (root / "logs" / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("fault", ["bindings", "result"])
def test_boot_recovery_claims_no_resume_until_its_resolver_is_admitted(pool, tmp_path, monkeypatch, fault):  # noqa: F811
    from supervisor import update_merge

    repo = _boot_recovery(pool, tmp_path, monkeypatch)
    if fault == "bindings":
        broken = registry._bindings_path(pool.root)
        broken.parent.mkdir(parents=True, exist_ok=True)
        original = '{"bindings": {}}'.encode()
        broken.write_text("{torn", encoding="utf-8")
    else:  # the prior resolver's own result is unreadable: it may have run
        write_task_result(pool.root, RESOLVER, "interrupted", admitted_dispatch="possible")
        original = result_signature(pool.root, RESOLVER)
        unreadable = unreadable_result(monkeypatch, RESOLVER)
    assert update_merge.finalize_managed_update_on_boot(supervisor_ready=True) == {
        "finalized": False, "resumed": False, "resolution_attempts": 1}
    types = _supervisor_types(pool.root)
    assert "managed_update_assisted_resume_unadmitted" in types and "managed_update_assisted_resumed" not in types
    stored = update_merge.read_update_tx()
    assert stored["phase"] == "assisted_resolution" and stored["resolution_attempts"] == 1  # retried next boot
    assert update_merge._merge_head_sha() == stored["target_sha"]  # never rolled back
    assert (repo / "a.txt").read_text(encoding="utf-8") == "the resolver's precious resolution\n"
    assert not queue.PENDING and not queue.RUNNING
    if fault == "bindings":
        broken.write_bytes(original)
    else:
        unreadable["on"] = False
        assert result_signature(pool.root, RESOLVER) == original
    assert update_merge.finalize_managed_update_on_boot(supervisor_ready=True) == {
        "finalized": False, "resumed": True, "resolution_attempts": 2}
    assert [row["id"] for row in queue.PENDING] == [RESOLVER]
    workers.assign_tasks()
    assert _drain(pool.slot) == [RESOLVER]


@pytest.mark.parametrize("veto", [None, "terminal", "foreign_tx"])
def test_boot_recovery_resumes_a_resolver_restored_from_its_claimed_handoff_once(pool, tmp_path, monkeypatch, veto):  # noqa: F811
    """R1: the server dies after the handoff, before its assign snapshot. Restore holds the
    claimed 'possible' row beside the worker's readable result; the NEXT real boot recovery's
    same-id resume then releases that exact row once. Stop, a terminal or unreadable result,
    a scope change and another transaction each keep it from ever being sent again."""
    from ouroboros.cancel_intents import request_cancel
    from supervisor import update_merge

    _boot_recovery(pool, tmp_path, monkeypatch)
    assert update_merge.finalize_managed_update_on_boot(supervisor_ready=True)["resumed"] is True
    receipt = load_task_result(pool.root, RESOLVER, strict=True)["host_admission"]
    claimed, put = [], pool.slot.in_q.put
    def capture_claim(row, *args, **kwargs):
        claimed.append({name: queue.QUEUE_SNAPSHOT_PATH.with_name(name).read_bytes()
                        for name in ("queue_snapshot.json", "queue_dispatch.json", "queue.alive")})
        return put(row, *args, **kwargs)
    pool.slot.in_q.put = capture_claim
    workers.assign_tasks()
    assert _drain(pool.slot) == [RESOLVER] and len(claimed) == 1
    write_task_result(pool.root, RESOLVER, "running", started_at="2026-01-01T00:00:00Z")  # the worker ran
    queue.RUNNING.clear()
    pool.slot.busy_task_id = None
    queue.PENDING.clear()
    for name, data in claimed[0].items():  # the post-handoff publication never landed
        queue.QUEUE_SNAPSHOT_PATH.with_name(name).write_bytes(data)
    assert queue.restore_pending_from_snapshot() == 1
    [held] = queue.PENDING
    assert held["_project_admission_restore_hold"] and held["admitted_dispatch"] == "possible"
    workers.assign_tasks()
    assert not _drain(pool.slot)  # restore alone never replays a possibly dispatched row
    original = result_signature(pool.root, RESOLVER)
    unreadable = {"on": False}
    if veto == "stop":
        request_cancel(pool.root, RESOLVER, reason="owner stopped")
    elif veto == "terminal":  # shutdown custody settled it: boot recovery mints a fresh id
        write_task_result(pool.root, RESOLVER, "cancelled", cancel_origin={"source": "snapshot_restore",
                                                                           "reason": "server_shutdown"})
    elif veto == "torn":
        unreadable = unreadable_result(monkeypatch, RESOLVER)
    elif veto == "bound":  # the unscoped resolver's assignment changed after its admission
        registry.create_project(pool.root, "room")
        registry.bind_task_to_project(pool.root, RESOLVER, "room", origin={"absent": "system"})
    resumed = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)["resumed"]
    assert resumed is (veto != "torn")
    if veto == "foreign_tx":  # the transaction that granted this resume is no longer live
        assert update_merge.clear_update_tx()
    for _pass in range(3):
        workers.assign_tasks()
    fresh = update_merge.read_update_tx().get("task_id")
    assert _drain(pool.slot) == {None: [RESOLVER], "terminal": [fresh]}.get(veto, [])  # at most once
    waiting = [row["id"] for row in queue.PENDING if row.get("_project_admission_restore_hold")]
    assert waiting == ([RESOLVER] if veto in {"torn", "foreign_tx"} else [])
    if veto == "torn":  # unknown dispatch waits; the next boot's resume, not readability, releases it
        unreadable["on"] = False
        assert result_signature(pool.root, RESOLVER) == original
        workers.assign_tasks()
        assert not _drain(pool.slot) and queue.PENDING[0]["_project_admission_restore_hold"]
        assert update_merge.finalize_managed_update_on_boot(supervisor_ready=True)["resumed"] is True
        workers.assign_tasks()
        workers.assign_tasks()
        assert _drain(pool.slot) == [RESOLVER] and not queue.PENDING
        return
    stored = load_task_result(pool.root, RESOLVER, strict=True)
    assert stored["status"] == {None: "running", "stop": "cancelled", "terminal": "cancelled",
                                "bound": "failed", "foreign_tx": "running"}[veto]
    if veto is None:  # the same accepted row continues: 'possible' kept, no receipt minted
        assert queue.RUNNING[RESOLVER]["task"]["admitted_dispatch"] == "possible"
        assert "_managed_update_resume" not in queue.RUNNING[RESOLVER]["task"]
        assert stored["host_admission"] == receipt and stored["admitted_dispatch"] == "possible"


@pytest.mark.parametrize("bindings", ["torn", "healthy"])
def test_assisted_apply_reports_started_only_for_an_admitted_resolver(pool, monkeypatch, bindings):  # noqa: F811
    import ouroboros.reviewer_slot_config as reviewer_slot_config
    import supervisor.worker_chat_lane as worker_chat_lane
    from ouroboros.gateway import control
    from supervisor import git_ops, state, update_merge

    base = "a" * 40

    def capture(cmd):
        if "--abbrev-ref" in cmd:
            return 0, "ouroboros", ""
        if "MERGE_HEAD" in cmd:
            return 1, "", ""
        return (0, base, "") if cmd[:3] == ["git", "rev-parse", "--verify"] else (0, "", "")

    txs, rollbacks = [], []
    monkeypatch.setattr(reviewer_slot_config, "review_pool_slots", lambda **_kw: [])
    monkeypatch.setattr(git_ops, "BRANCH_DEV", "ouroboros")
    monkeypatch.setattr(git_ops, "git_capture", capture)
    monkeypatch.setattr(git_ops, "_create_rescue_snapshot", lambda *_a, **_k: None)
    monkeypatch.setattr(git_ops, "_collect_repo_sync_state", lambda: {})
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 10.0)
    monkeypatch.setattr(update_merge, "write_update_tx", lambda tx: txs.append(dict(tx)))
    monkeypatch.setattr(update_merge, "ensure_assisted_resolver_ready", lambda _sha: True)
    monkeypatch.setattr(update_merge, "materialize_assisted_merge_live", lambda *_a: (True, "ok", "m0tree"))
    monkeypatch.setattr(update_merge, "rollback_managed_update",
                        lambda reason, **_k: rollbacks.append(reason) or (True, "rolled back"))
    monkeypatch.setattr(worker_chat_lane, "preload_owner_control_path", lambda: [])
    monkeypatch.setattr(workers, "close_repo_writer_admission", lambda _reason: None)
    monkeypatch.setattr(control, "_respawn_workers_after_failed_update", lambda: None)
    if bindings == "torn":
        path = registry._bindings_path(pool.root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{torn", encoding="utf-8")
    plan = {"base_sha": base, "target_sha": "b" * 40, "local_snapshot": base, "kind": "conflicting",
            "code_conflict_paths": ["ouroboros/config.py"], "doc_conflict_paths": []}
    response = control._start_assisted_merge_fenced(
        plan, {"phase": "stashing_local_work", "stash_sha": "", "local_work_carrier": "none"})
    body = json.loads(response.body)
    task_id = txs[-1]["task_id"]
    # The fresh transaction records its submission ahead of the one admission attempt;
    # a proven refusal appended nothing, so it records the id unsubmitted again.
    assert [tx.get("resolver_submitted_id") for tx in txs] == ["", "", task_id] + [""] * (bindings == "torn")
    if bindings == "torn":
        assert response.status_code == 409 and body["rolled_back"] is True and "status" not in body
        assert rollbacks == ["assisted_worker_start_failed"]
        assert not queue.PENDING and load_task_result(pool.root, task_id) is None
        assert _unadmitted(pool.root) == ["project_routing_fence_lookup_failed"]
    else:
        assert response.status_code == 200 and body["status"] == "assisted_started" and body["task_id"] == task_id
        assert not rollbacks
        [row] = queue.PENDING
        assert row["id"] == task_id and row["admitted_dispatch"] == "none"  # a fresh id's provable first
        assert load_task_result(pool.root, task_id, strict=True)["status"] == "scheduled"
        workers.assign_tasks()
        assert _drain(pool.slot) == [task_id]


def _child_of(host, tmp_path, parent):  # noqa: F811
    """A real Project admission queued as the child of ``parent`` (whose shutdown state varies)."""
    child = accepted(host, tmp_path, tid="child")
    child.update(parent_task_id="parent", root_task_id="parent", delegation_role="subagent")
    write_task_result(host.root, "parent", "running" if parent == "interrupted" else "completed", chat_id=1)
    assert queue.persist_queue_snapshot()
    if parent == "interrupted":  # the parent was RUNNING when the server stopped
        snap = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        snap["running"] = [{"id": "parent", "task": {"id": "parent", "chat_id": 1}}]
        queue.QUEUE_SNAPSHOT_PATH.write_text(json.dumps(snap), encoding="utf-8")


@pytest.mark.parametrize("parent", ["interrupted", "unreadable", "completed"])
def test_held_project_child_never_starts_behind_an_interrupted_parent(host, tmp_path, monkeypatch, parent):  # noqa: F811
    _child_of(host, tmp_path, parent)
    parent_unreadable = unreadable_result(monkeypatch, "parent") if parent == "unreadable" else {"on": False}
    original = result_signature(host.root, "child")
    child_unreadable = unreadable_result(monkeypatch, "child")  # the child's own receipt is unknown at restore
    host.pending.clear()
    queue.restore_pending_from_snapshot()
    [held] = host.pending
    assert held["_project_admission_restore_hold"] and not held.get("_terminalization_retry")
    sent = worker(host, monkeypatch)
    child_unreadable["on"] = False
    assert result_signature(host.root, "child") == original
    if parent == "completed":
        resume_after_app_stop(host, "child")
    workers.assign_tasks()
    workers.assign_tasks()
    assert [row["id"] for row in sent] == (["child"] if parent == "completed" else [])
    if parent == "interrupted":  # the restore's shutdown custody, never a later revival
        assert load_task_result(host.root, "child")["status"] == "cancelled" and not host.pending
    elif parent == "unreadable":  # unknown lineage keeps the same accepted row waiting
        assert host.pending[0]["_project_admission_restore_hold"]
        assert not host.pending[0].get("_terminalization_retry")
        parent_unreadable["on"] = False
        resume_after_app_stop(host, "child")
        workers.assign_tasks()
        workers.assign_tasks()
        assert [row["id"] for row in sent] == ["child"]


@pytest.mark.parametrize("parent", ["completed", "shutdown", "unreadable"])
def test_readable_project_child_waits_for_its_unreadable_parent(host, tmp_path, monkeypatch, parent):  # noqa: F811
    """Only the parent is unknown at restore: the child's receipt and the bindings are healthy."""
    _child_of(host, tmp_path, "completed")
    parent_unreadable = unreadable_result(monkeypatch, "parent")
    host.pending.clear()
    queue.restore_pending_from_snapshot()
    [held] = host.pending
    assert held["id"] == "child" and held["_project_admission_restore_hold"]
    assert not held.get("_terminalization_retry")  # neither interruption nor permission
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert not sent and load_task_result(host.root, "child")["status"] == "scheduled"
    if parent == "completed":
        parent_unreadable["on"] = False
        resume_after_app_stop(host, "child")
    elif parent == "shutdown":  # an earlier boot handed the parent to shutdown custody
        parent_unreadable["on"] = False
        delete_result(host.root, "parent")
        write_task_result(host.root, "parent", "cancelled", chat_id=1,
                          cancel_origin={"source": "snapshot_restore", "reason": "server_shutdown"})
    for _pass in range(3):
        workers.assign_tasks()
    assert [row["id"] for row in sent] == (["child"] if parent == "completed" else [])  # exactly once
    if parent == "shutdown":
        assert load_task_result(host.root, "child")["status"] == "cancelled" and not host.pending
    elif parent == "unreadable":
        assert host.pending[0]["_project_admission_restore_hold"] and not host.attempts


def _fault_parent(host, monkeypatch, fault):  # noqa: F811
    """The parent's evidence fails; returns the switch that makes it readable again."""
    if fault == "missing":
        delete_result(host.root, "parent")
    elif fault == "torn":
        return unreadable_result(monkeypatch, "parent")
    elif fault == "never_admitted":  # a refusal receipt cannot be the parent that spawned it
        delete_result(host.root, "parent")
        write_task_result(host.root, "parent", "failed", chat_id=1, admission_outcome="never_admitted")
    elif fault == "shutdown":  # an earlier boot handed the parent to shutdown custody
        delete_result(host.root, "parent")
        write_task_result(host.root, "parent", "cancelled", chat_id=1,
                          cancel_origin={"source": "snapshot_restore", "reason": "server_shutdown"})
    return {"on": False}


def _passes(count=4, busy=""):
    for _pass in range(count):  # each dispatched row finishes before the next pass
        workers.WORKERS[0].busy_task_id = busy or None  # busy: holds are revalidated, nothing starts
        workers.assign_tasks()
        queue.RUNNING.clear()
        workers.WORKERS[0].busy_task_id = None


# A queued parent whose own result tears AFTER restore is terminalized by assignment's
# result-authority owner, so the torn queued case is exercised at restore only.
@pytest.mark.parametrize("where,fault,phase", chosen_rows([
    *((where, fault, phase) for phase in ("restore", "release") for where, fault in (
        ("queued", None), ("queued", "missing"), ("queued", "shutdown"),
        ("external", "missing"), ("external", "never_admitted"))),
    ("queued", "torn", "restore")], 0, 9, 10))
def test_healthy_child_waits_for_an_unknown_parent_inside_or_outside_the_snapshot(
        host, tmp_path, monkeypatch, where, fault, phase):  # noqa: F811
    """A child's own evidence is healthy; only its parent's is not. A parent queued beside
    it or known only by its result, whose result is missing, unreadable or a refusal,
    keeps the child waiting at restore and at hold release; a parent the shutdown stopped
    cancels it; a readable parent lets each id start exactly once."""
    if where == "queued":
        accepted(host, tmp_path, tid="parent")  # restored beside the child, with its own receipt
    child = accepted(host, tmp_path, tid="child")
    child.update(parent_task_id="parent", root_task_id="parent", delegation_role="subagent")
    if where == "external":
        write_task_result(host.root, "parent", "completed", chat_id=1)
    assert queue.persist_queue_snapshot()
    parent_document = load_task_result(host.root, "parent")
    child_signature = result_signature(host.root, "child")
    parent_unreadable = {"on": False}
    if phase == "restore":
        parent_unreadable = _fault_parent(host, monkeypatch, fault)
    else:  # the child's own receipt is what is unknown at restore; its parent is healthy
        child_unreadable = unreadable_result(monkeypatch, "child")
    host.pending.clear()
    queue.restore_pending_from_snapshot()
    sent = worker(host, monkeypatch)
    if phase == "release":
        assert next(row for row in host.pending if row["id"] == "child")["_project_admission_restore_hold"]
        parent_unreadable = _fault_parent(host, monkeypatch, fault)  # the parent's evidence fails before release
        child_unreadable["on"] = False
        assert result_signature(host.root, "child") == child_signature
        _passes(1, busy="busy")
    rows = {row["id"]: row for row in host.pending}
    if fault == "shutdown":  # the restore's shutdown custody, never a later revival
        _passes()
        assert not sent and load_task_result(host.root, "child")["status"] == "cancelled"
        assert "child" not in {row["id"] for row in host.pending}
        return
    if fault:  # neither interruption nor permission: the same accepted row waits
        assert rows["child"]["_project_admission_restore_hold"]
        assert not rows["child"].get("_terminalization_retry")
        # A queued parent restored healthy is itself dispatchable; it must not start here.
        _passes(busy="busy" if where == "queued" and phase == "release" else "")
        assert "child" not in [row["id"] for row in sent]
        assert load_task_result(host.root, "child")["status"] == "scheduled"
        if fault == "never_admitted":
            assert host.pending[-1]["_project_admission_restore_hold"] and not host.attempts
            return
        parent_unreadable["on"] = False  # the parent's evidence returns
        if not result_exists(host.root, "parent"):
            seed_result(host.root, "parent", parent_document)
    else:  # a readable parent satisfies lineage; explicit Resume still releases the app-stop hold
        assert not rows["child"].get("_project_admission_restore_hold")
    resume_after_app_stop(host, *("parent", "child") if where == "queued" else ("child",))
    _passes()
    assert [row["id"] for row in sent] == (["parent", "child"] if where == "queued" else ["child"])
    assert not host.pending and not host.attempts


@pytest.mark.parametrize("veto", [None, "forged", "revoked"])
def test_exact_pause_of_held_project_row_continues_only_under_its_resume_grant(host, tmp_path, monkeypatch, veto):  # noqa: F811
    from ouroboros import budget_pause
    from tests._budget_pause_exact_helpers import _pause

    prepared = copy.deepcopy(accepted(host, tmp_path))
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["held"]
    running = queue.RUNNING.pop("held")["task"]
    workers.WORKERS[0].busy_task_id = None
    _pause(host.root, monkeypatch, task_id="held")  # the dispatched run paused mid-flight
    budget_pause.end_dispatch_fence("held")
    row = budget_pause.budget_pause_row(host.root, "held")
    host.pending.append({**running, "_budget_pause": budget_pause.exact_pause_marker(row, default_root="held")})
    budget_pause.set_budget_pause(host.root, "held", {**row, "state": budget_pause.STATE_PAUSED})
    path, original = restore_unreadable(host)
    path.write_bytes(original)
    workers.assign_tasks()
    assert len(sent) == 1 and host.pending[0]["_project_admission_restore_hold"]  # paused, never replayed
    assert queue.resume_budget_paused_task("held")["ok"] is True
    [held] = host.pending
    if veto == "forged":
        held["_budget_pause_resume"]["grant_id"] = "forged"
    elif veto == "revoked":
        live = budget_pause.budget_pause_row(host.root, "held")
        budget_pause.set_budget_pause(host.root, "held", {**live, "grant": {**live["grant"], "revoked_at": "now"}})
    workers.assign_tasks()
    workers.assign_tasks()
    assert [row["id"] for row in sent] == (["held"] if veto else ["held", "held"])
    assert load_task_result(host.root, "held")["status"] != "failed" and not host.attempts
    if veto:
        assert host.pending[0]["_project_admission_restore_hold"]
    else:
        assert sent[1]["_budget_pause_resume"]["grant_id"] and "_budget_pause" not in sent[1]
        assert sent[1]["_project_admission"] == prepared["_project_admission"]
        assert sent[1]["workspace_root"] == prepared["workspace_root"]


def _consciousness_review(monkeypatch, window):
    from ouroboros import consciousness_allowance
    from ouroboros.consciousness_authority import consciousness_origin_metadata

    entered, release, reads = threading.Event(), threading.Event(), []

    def contended_ledger(*_args, **_kwargs):
        entered.set()
        assert release.wait(10)
        reads.append(True)
        return window

    monkeypatch.setattr(consciousness_allowance, "allowance_window", contended_ledger)
    origin = consciousness_origin_metadata({"initiator": "consciousness", "consciousness_autonomy": "full"})
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(queue.queue_deep_self_review_task, "consciousness:review",
                                 force=True, chat_id=1, origin=origin)
        assert entered.wait(10)
        free = queue._queue_lock.acquire(timeout=2)
        if free:
            queue._queue_lock.release()
        release.set()
        return future.result(timeout=30), free, reads


@pytest.mark.parametrize("status", ["available", "exhausted"])
def test_consciousness_host_producer_reads_its_allowance_off_the_queue_lock(host, monkeypatch, status):  # noqa: F811
    from ouroboros.consciousness_allowance import STATUS_AVAILABLE

    notices = []
    monkeypatch.setattr(queue, "send_with_budget", lambda _chat, text, **k: notices.append((text, k)))
    window = {"status": STATUS_AVAILABLE if status == "available" else "exhausted", "limit_usd": 5.0,
              "settled_usd": 0.0 if status == "available" else 5.0,
              "accounted_usd": 0.0 if status == "available" else 5.0, "remaining_usd": 5.0,
              "resets_at": "", "unknown_unmetered": 0}
    tid, free, reads = _consciousness_review(monkeypatch, window)
    assert free, "a contended usage ledger held the global queue lock"
    assert reads == [True]  # one exact admission read, still the money door
    if status == "available":
        assert [row["id"] for row in host.pending] == [tid]
        assert load_task_result(host.root, tid)["host_admission"]["status"] == "accepted"
    else:
        assert tid is None and not host.pending
        text, kwargs = notices[-1]
        assert text == "Deep self-review could not be queued."
        notice = kwargs["progress_meta"]["notice"]  # the allowance's typed cause is the refusal's detail
        assert notice["code"] == "deep_self_review_not_admitted"
        assert notice["details"].startswith("consciousness_allowance_exhausted")
