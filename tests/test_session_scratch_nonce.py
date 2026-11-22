"""New scratch uses a logical container identity, without migrating legacy directories.

Only synthetic allocations of this test run are changed or removed. Marker reads are confined,
bounded and read-only; an unknown marker never falls back to a matching device/inode. Whole-folder
restoration at the recorded address inherits the old holder and cleanup by the owner's chosen contract.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import stat
import subprocess
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ouroboros import owner_keep, task_custody, utils, work_domain
from ouroboros import session_scratch as scratch
from ouroboros.workspace_preflight import release_finished_scratch
from tests._store_shared import seed_result
from tests.test_execution_domain import posix_host
from tests.test_session_scratch_release import OVER, _container, _record
from tests.test_session_service_directories import (  # noqa: F401 -- confined temp parent
    GRANTED,
    _Catalog,
    _legacy_scratch,
    _prepare,
    _synthetic_temp_parent,
)


def _made(tmp_path, **kwargs):
    service = _prepare(tmp_path, _Catalog(GRANTED), **kwargs)
    assert service.scratch, service
    data = tmp_path / "data"
    return service, data, pathlib.Path(service.scratch), _record(data, key=kwargs.get("key", "snap-1"))


def _stamps(*paths):
    return [(path.lstat().st_mtime_ns, path.read_bytes()) for path in paths]


def test_allocation_publishes_only_after_the_nonce_and_empty_workspace_exist(tmp_path, monkeypatch):
    original, published = utils.atomic_write_json, []

    def publish(path, row, **kwargs):
        if pathlib.Path(path).name == "scratch.json":
            container = pathlib.Path(row["path"])
            assert row["schema_version"] == 2 and set(row) == {"schema_version", "path", "nonce"}
            assert len(row["nonce"]) == 32
            assert (container / scratch.SCRATCH_MARKER).read_bytes() == (row["nonce"] + "\n").encode("ascii")
            assert sorted(item.name for item in container.iterdir()) == [scratch.SCRATCH_MARKER, "w"]
            assert list((container / "w").iterdir()) == []
            published.append(str(container))
        return original(path, row, **kwargs)

    monkeypatch.setattr(utils, "atomic_write_json", publish)
    service, data, workspace, record = _made(tmp_path)
    container = _container(data)
    assert service.roots[0] == str(workspace) and workspace == container / "w"
    assert str(container) not in service.roots and str(record.parent) not in service.roots
    marker = container / scratch.SCRATCH_MARKER
    before = _stamps(record, marker)
    (workspace / "kept.txt").write_bytes(b"SAME-KEY")
    assert _prepare(tmp_path, _Catalog(GRANTED)) == service and published == [str(container)]
    assert (workspace / "kept.txt").read_bytes() == b"SAME-KEY" and _stamps(record, marker) == before
    assert scratch.recorded_scratch_roots(data) == [str(container)], "count the container once, not it and w"


def test_restored_container_at_the_same_address_inherits_keep_and_then_cleanup(tmp_path):
    service, data, workspace, record = _made(tmp_path)
    container = workspace.parent
    seed_result(data, "task-1", {"status": "completed"})
    (workspace / "before.txt").write_bytes(b"BEFORE-RESTORE")
    saved, original = tmp_path / "saved", tmp_path / "original"
    shutil.copytree(container, saved)
    container.rename(original)
    shutil.copytree(saved, container)
    assert container.lstat().st_ino != original.lstat().st_ino
    (workspace / "after.txt").write_bytes(b"AFTER-RESTORE")
    before = _stamps(record, container / scratch.SCRATCH_MARKER)
    assert scratch.recorded_scratch(record) == container
    assert _prepare(tmp_path, _Catalog(GRANTED)) == service
    assert _stamps(record, container / scratch.SCRATCH_MARKER) == before
    places, _unknown, pins = owner_keep.inventory(data, "task-1", {"status": "completed"})
    assert str(container) in places and str(workspace) not in places and pins[str(container)] == str(record)
    assert owner_keep.set_owner_keep(data, "task-1", keep=True, source={"basis": "test"})["keep"]
    assert [row["class"] for row in release_finished_scratch(data, OVER)["holds"]] == ["owner_keep"]
    assert (workspace / "after.txt").read_bytes() == b"AFTER-RESTORE"
    assert owner_keep.set_owner_keep(data, "task-1", keep=False, source={"basis": "test"})["keep"] is False
    assert release_finished_scratch(data, OVER)["released"]
    assert task_custody.remove_custody_trash(data) == {"removed": 1, "stuck": {}, "holds": []}
    assert not container.exists(), "the restored container, including later additions, has the old cleanup fate"
    assert (original / "w" / "before.txt").read_bytes() == b"BEFORE-RESTORE"
    assert saved.is_dir(), "no marker scan or adoption of another address"


def test_nonce_validation_is_independent_of_the_container_device_and_inode(tmp_path, monkeypatch):
    _service, data, workspace, record = _made(tmp_path)
    container, original = workspace.parent, work_domain.lstat
    before = _stamps(record, container / scratch.SCRATCH_MARKER)

    def renumbered(path):
        seen = original(path)
        return (SimpleNamespace(st_mode=seen.st_mode, st_dev=seen.st_dev + 1, st_ino=seen.st_ino + 1)
                if pathlib.Path(path) == container else seen)

    monkeypatch.setattr(work_domain, "lstat", renumbered)
    assert scratch.recorded_scratch(record) == container
    assert _prepare(tmp_path, _Catalog(GRANTED)).scratch == str(workspace)
    assert scratch.recorded_scratch_roots(data) == [str(container)]
    assert _stamps(record, container / scratch.SCRATCH_MARKER) == before


@pytest.mark.parametrize("fault", ["missing", "different", "oversized", "link", "directory", "fifo", "unreadable"])
def test_marker_failure_holds_without_inode_fallback_or_any_repair(tmp_path, monkeypatch, fault):
    _service, data, workspace, record = _made(tmp_path)
    container = workspace.parent
    marker = container / scratch.SCRATCH_MARKER
    row = json.loads(record.read_text(encoding="utf-8"))
    info = container.lstat()
    row["identity"] = [info.st_dev, info.st_ino]  # a matching legacy tuple must not rescue schema 2
    record.write_text(json.dumps(row), encoding="utf-8")
    if fault == "unreadable":
        from ouroboros import confined_files

        def refuse(*args, **kwargs):
            raise PermissionError("the marker cannot be opened")

        monkeypatch.setattr(confined_files, "open_regular_file", refuse)
    else:
        marker.unlink()
        if fault == "different":
            marker.write_bytes(b"f" * 32 + b"\n")
        elif fault == "oversized":
            marker.write_bytes((row["nonce"] + "\n").encode() + b"x" * 4096)
        elif fault == "link":
            target = tmp_path / "marker-target"
            target.write_bytes((row["nonce"] + "\n").encode())
            try:
                marker.symlink_to(target)
            except (OSError, NotImplementedError) as exc:
                pytest.skip(f"symlinks unavailable: {exc}")
        elif fault == "directory":
            marker.mkdir()
        elif fault == "fifo":
            if not hasattr(os, "mkfifo"):
                pytest.skip("FIFOs unavailable")
            os.mkfifo(marker)
    (workspace / "payload.txt").write_bytes(b"MUST-STAY")
    before, stamp = record.read_bytes(), container.lstat().st_mtime_ns
    with pytest.raises(scratch.ScratchIdentityMismatch) as refused:
        scratch.recorded_scratch(record)
    assert refused.value.basis == "marker"
    again = _prepare(tmp_path, _Catalog(GRANTED))
    assert not again.roots and not again.scratch and "not the recorded scratch" in again.limitation
    [held] = scratch.release_session_scratch(record.parents[2])
    assert held["class"] == "identity_unverified" and held["path"] == str(container)
    [listed] = scratch.scratch_records(data)
    assert not listed["valid"] and not listed["missing"] and listed["hold"] == "identity_unverified"
    assert scratch.recorded_scratch_roots(data) == []
    assert record.read_bytes() == before and container.lstat().st_mtime_ns == stamp
    assert (workspace / "payload.txt").read_bytes() == b"MUST-STAY"


@pytest.mark.parametrize("nonce", [None, "", "x" * 32, "a" * 31, 2, True])
def test_malformed_nonce_never_becomes_a_legacy_record_even_with_matching_inodes(tmp_path, nonce):
    _service, _data, workspace, record = _made(tmp_path)
    row = json.loads(record.read_text(encoding="utf-8"))
    seen = workspace.parent.lstat()
    row.update(nonce=nonce, identity=[seen.st_dev, seen.st_ino])
    record.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="records no scratch nonce"):
        scratch.recorded_scratch(record)
    assert _prepare(tmp_path, _Catalog(GRANTED)).roots == () and workspace.is_dir()


def test_missing_workspace_is_recreated_only_after_container_validation(tmp_path):
    _service, _data, workspace, record = _made(tmp_path)
    workspace.rmdir()
    container, marker = workspace.parent, workspace.parent / scratch.SCRATCH_MARKER
    before, stamp = _stamps(record, marker), container.lstat().st_mtime_ns
    assert scratch.recorded_scratch(record) == container and not workspace.exists()
    assert _stamps(record, marker) == before and container.lstat().st_mtime_ns == stamp
    assert _prepare(tmp_path, _Catalog(GRANTED)).scratch == str(workspace)
    assert workspace.is_dir() and list(workspace.iterdir()) == [] and _stamps(record, marker) == before


@pytest.mark.parametrize("replacement", ["link", "file"])
def test_existing_workspace_must_be_a_real_directory_not_a_grant_through_a_link(tmp_path, replacement):
    _service, _data, workspace, record = _made(tmp_path)
    workspace.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel").write_bytes(b"FOREIGN")
    if replacement == "link":
        try:
            workspace.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlinks unavailable: {exc}")
    else:
        workspace.write_bytes(b"not a workspace")
    before = record.read_bytes()
    service = _prepare(tmp_path, _Catalog(GRANTED))
    assert not service.roots and not service.scratch and "not a real scratch workspace" in service.limitation
    assert record.read_bytes() == before and (outside / "sentinel").read_bytes() == b"FOREIGN"


def test_legacy_same_key_never_upgrades_even_when_matching_and_mismatch_never_reallocates(tmp_path):
    _service, data, _workspace, record = _made(tmp_path)
    legacy = _legacy_scratch(record)
    before = _stamps(record)
    assert _prepare(tmp_path, _Catalog(GRANTED)).scratch == str(legacy) and _stamps(record) == before
    row = json.loads(record.read_text(encoding="utf-8"))
    row["identity"][0] += 1
    record.write_text(json.dumps(row), encoding="utf-8")
    before, names = _stamps(record), sorted(legacy.parent.iterdir())
    assert _prepare(tmp_path, _Catalog(GRANTED)).roots == ()
    assert _stamps(record) == before and sorted(legacy.parent.iterdir()) == names
    assert json.loads(record.read_text())["schema_version"] == 1
    assert scratch.scratch_records(data)[0]["hold"] == "identity_unverified"


@pytest.mark.parametrize("phase", ["marker", "workspace", "record"])
def test_pregrant_failure_rolls_back_only_this_allocation_and_never_grants_it(tmp_path, monkeypatch, phase):
    sibling, _data, sibling_workspace, _record_path = _made(tmp_path, key="sibling")
    (sibling_workspace / "sentinel").write_bytes(b"SIBLING")
    parent = sibling_workspace.parent.parent
    before = sorted(parent.iterdir())
    if phase == "marker":
        original = pathlib.Path.open

        def fail(path, *args, **kwargs):
            if path.name == scratch.SCRATCH_MARKER:
                raise OSError("marker write refused")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "open", fail)
    elif phase == "workspace":
        original = pathlib.Path.mkdir

        def fail(path, *args, **kwargs):
            if path.name == "w":
                raise OSError("workspace mkdir refused")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "mkdir", fail)
    else:
        original = utils.atomic_write_json

        def fail(path, row, **kwargs):
            if pathlib.Path(path).name == "scratch.json":
                raise OSError("record write refused")
            return original(path, row, **kwargs)

        monkeypatch.setattr(utils, "atomic_write_json", fail)
    refused = _prepare(tmp_path, _Catalog(GRANTED))
    assert not refused.roots and not refused.scratch and "refused" in refused.limitation
    assert sorted(parent.iterdir()) == before
    assert not _record(tmp_path / "data").exists()
    assert not (tmp_path / "data" / "state/session_scratch/task-1/snap-1.json").exists()
    assert sibling.roots and (sibling_workspace / "sentinel").read_bytes() == b"SIBLING"


def test_managed_marker_reads_descend_from_the_mapping_and_never_use_a_link_fallback(tmp_path, monkeypatch):
    _service, _data, workspace, record = _made(tmp_path)
    domain = SimpleNamespace(work_roots=((str(workspace.parent.parent), str(workspace.parent.parent)),),
                             staging_root=str(workspace.parent.parent))
    monkeypatch.setattr(work_domain, "_domain", lambda: domain)

    def forbidden(*args, **kwargs):
        raise AssertionError("marker identity must not fall back to an in-work cat or generic read")

    monkeypatch.setattr(work_domain, "read_in_work", forbidden)
    monkeypatch.setattr(work_domain, "read_bytes", forbidden)
    from tests.test_work_domain import service_effects_on

    with service_effects_on(workspace.parent.parent, allowed_opens=(str(workspace.parent.parent),)) as seen:
        assert scratch.recorded_scratch(record) == workspace.parent
    assert not seen, seen
    marker = workspace.parent / scratch.SCRATCH_MARKER
    target = tmp_path / "service-only-marker"
    target.write_bytes(marker.read_bytes())
    marker.unlink()
    marker.symlink_to(target)
    with pytest.raises(scratch.ScratchIdentityMismatch):
        scratch.recorded_scratch(record)


def test_marker_read_is_bounded_and_an_oversized_marker_is_never_read(tmp_path, monkeypatch):
    from ouroboros import confined_files

    _service, _data, workspace, record = _made(tmp_path)
    original, reads = confined_files.open_regular_file, []

    def opened(*args):
        stream, info = original(*args)

        class Bounded:
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                stream.close()

            def read(self, size):
                assert 0 < size <= 34
                reads.append(size)
                return stream.read(size)

        return Bounded(), info

    monkeypatch.setattr(confined_files, "open_regular_file", opened)
    assert scratch.recorded_scratch(record) == workspace.parent and reads == [34]
    reads.clear()
    with (workspace.parent / scratch.SCRATCH_MARKER).open("ab") as marker:
        marker.write(b"oversized")
    with pytest.raises(scratch.ScratchIdentityMismatch):
        scratch.recorded_scratch(record)
    assert not reads, "size mismatch refuses before any marker content is read"


@pytest.mark.parametrize("entry", ["container", "workspace"])
def test_new_identity_never_grants_a_reparse_directory(tmp_path, monkeypatch, entry):
    _service, _data, workspace, record = _made(tmp_path)
    target = workspace.parent if entry == "container" else workspace
    original = work_domain.lstat

    def reparse(path):
        seen = original(path)
        return SimpleNamespace(st_mode=seen.st_mode, st_file_attributes=0x400) if pathlib.Path(path) == target else seen

    monkeypatch.setattr(work_domain, "lstat", reparse)
    before = record.read_bytes()
    service = _prepare(tmp_path, _Catalog(GRANTED))
    assert not service.roots and not service.scratch and record.read_bytes() == before and workspace.is_dir()


@pytest.mark.skipif(os.name != "posix", reason="POSIX AF_UNIX consumer; Windows uses its own IPC layout")
def test_a_real_97_byte_socket_binds_below_the_short_granted_workspace(tmp_path, monkeypatch):
    import socket

    from ouroboros import test_environment
    from ouroboros.test_evidence import evidence_directory
    from tests.test_session_service_directories import _PLATFORM_TEMP_PARENT

    monkeypatch.setattr(test_environment, "session_scratch_parent", _PLATFORM_TEMP_PARENT)
    service, _data, workspace, record = _made(tmp_path)
    folder = workspace / "ob-12345678.noindex" / "b0" / "t" / "pymp-12345678"
    address = folder / ("listener-" + "s" * (97 - len(os.fsencode(str(folder / "listener-")))))
    assert len(os.fsencode(str(address))) == 97
    folder.mkdir(parents=True)
    endpoint = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        endpoint.bind(str(address))
        facts = {"path_bytes": len(os.fsencode(str(address))), "bound": endpoint.getsockname(),
                 "grant": service.roots[0], "container": str(workspace.parent)}
        evidence = evidence_directory(tmp_path, parent=os.environ.get("OUROBOROS_TEST_EVIDENCE_ROOT")
                                      or tmp_path / "evidence")
        (evidence / "scratch-socket-bind.json").write_text(json.dumps(facts), encoding="utf-8")
        print(json.dumps(facts))
    finally:
        endpoint.close()
        scratch.release_session_scratch(record.parents[2])  # only this test's recorded container


@pytest.mark.parametrize("phase", ["container", "marker", "workspace"])
def test_an_unsettled_managed_allocation_never_retries_or_rolls_back_a_live_writer(tmp_path, monkeypatch, phase):
    """The transport is a synthetic file writer; the real preparation/refusal seam judges its unknown end."""
    work = tmp_path / "managed"
    work.mkdir()
    domain = SimpleNamespace(work_roots=((str(work), str(work)),), staging_root=str(work))
    monkeypatch.setattr(work_domain, "_domain", lambda: domain)
    made, injected = [], []

    def owned(path):
        """This fake domain's membership by path parts, on every host. The product compares POSIX spellings
        (``work_domain._rest``), which a Windows path never has: ``_workspace`` would bypass the fake writer."""
        path = pathlib.Path(os.path.abspath(path))
        return path == work or work in path.parents

    def unsettled(path):
        injected.append(path)
        raise work_domain.WorkDomainUnsettled("synthetic writer end unconfirmed")

    def mkdir(path, **_kwargs):
        path = pathlib.Path(path)
        path.mkdir()
        made.append(path)
        if (phase == "container" and path.name.startswith("obs-")) or (phase == "workspace" and path.name == "w"):
            unsettled(path)

    def write(path, data, **kwargs):
        pathlib.Path(path).write_bytes(data)
        if phase == "marker":
            unsettled(pathlib.Path(path))

    def forbidden(*args, **kwargs):
        raise AssertionError("an unsettled writer must not be removed before custody settles it")

    monkeypatch.setattr(work_domain, "is_work", owned)
    monkeypatch.setattr(work_domain, "make_dirs", lambda path, **_kwargs: pathlib.Path(path).mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(work_domain, "make_dir", mkdir)
    monkeypatch.setattr(work_domain, "write_file", write)
    monkeypatch.setattr(work_domain, "remove", forbidden)
    monkeypatch.setattr(work_domain, "remove_empty_dir", forbidden)
    refused = _prepare(tmp_path, _Catalog(GRANTED), access="full")
    assert len(injected) == 1, f"the {phase} fault must fire exactly once, through the fake writer: {injected}"
    assert not refused.roots and not refused.scratch and "unconfirmed" in refused.limitation
    [container] = [path for path in made if path.name.startswith("obs-")]
    assert injected == [{"container": container, "marker": container / scratch.SCRATCH_MARKER,
                         "workspace": container / "w"}[phase]], "the fault fired at the chosen phase's own path"
    assert container.is_dir(), "the single new allocation stays, not eight retries or a premature deletion"
    assert not _record(tmp_path / "data").exists()
    if phase != "container":
        assert (container / scratch.SCRATCH_MARKER).is_file()


def _file_work_transport(tmp_path, monkeypatch, *, phase="", settled=True):
    """Keep run_in_work/_effect real; only AttachedLaunch/Popen are file-only doubles, never OS children."""
    from ouroboros import execution_domain, platform_layer, tool_custody
    from ouroboros.tools import shell_process

    work = tmp_path / "managed"
    work.mkdir()
    domain = SimpleNamespace(work_roots=((str(work), str(work)),), staging_root=str(work))
    monkeypatch.setattr(work_domain, "_domain", lambda: domain)
    calls, launches, reaped = [], [], []

    class Proc:
        def __init__(self, words, timed_out):
            self.args, self.timed_out, self.returncode = words, timed_out, None

        def communicate(self, timeout=None):
            if self.timed_out:
                raise subprocess.TimeoutExpired(self.args, timeout, output=b"partial writer", stderr=b"deadline")
            self.returncode = 0
            return b"", b""

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    class Launch:
        def __init__(self, argv):
            self.argv, self.env, self.proc, self.settlements = argv, {}, None, []
            launches.append(self)

        def require_cli(self):
            pass

        def adopt(self):
            pass

        def settle(self):
            value = settled if self.proc.timed_out else True
            self.settlements.append(value)
            return value

    def popen(_factory, argv, **kwargs):
        words = list(argv[4:])  # run_in_work's literal `sh -c 'exec env "$@"' sh <operation>`
        calls.append(words)
        if words[0] == "mkdir":
            path = pathlib.Path(words[-1])
            made = not path.exists()
            path.mkdir(parents="-p" in words, exist_ok="-p" in words)
            if made and "-m" in words:
                path.chmod(int(words[words.index("-m") + 1], 8))
            effect = "container" if path.name.startswith("obs-") else "workspace" if path.name == "w" else "parent"
        elif words[0] == "sh":
            target, temporary, mode = pathlib.Path(words[4]), pathlib.Path(words[5]), words[6]
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_bytes(kwargs["stdin"].read())
            if mode:
                temporary.chmod(int(mode, 8))
            temporary.replace(target)
            effect = "marker"
        elif words[0] == "rmdir":
            pathlib.Path(words[-1]).rmdir()
            effect = "remove"
        elif words[:2] == ["rm", "-f"]:
            pathlib.Path(words[-1]).unlink(missing_ok=True)
            effect = "remove"
        else:
            raise AssertionError(f"not a synthetic file effect: {words}")
        proc = Proc(words, effect == phase)
        launches[-1].proc = proc
        return proc

    @contextmanager
    def tracked():
        yield lambda _proc: None

    monkeypatch.setattr(execution_domain, "attached_launch", lambda argv, *_a: Launch(argv))
    monkeypatch.setattr(platform_layer, "subprocess_new_group_kwargs", lambda: {})
    monkeypatch.setattr(tool_custody, "external_handoff", lambda _kind: lambda: None)
    monkeypatch.setattr(shell_process, "tracked_start", tracked)
    monkeypatch.setattr(shell_process, "contained_popen", popen)
    monkeypatch.setattr(shell_process, "_kill_process_group", lambda proc: setattr(proc, "returncode", -9))
    monkeypatch.setattr(shell_process, "retire_tracked", lambda proc: reaped.append(proc))
    return work, calls, launches, reaped


@posix_host
@pytest.mark.parametrize("phase", ["container", "marker", "workspace"])
@pytest.mark.parametrize("settled", [False, True], ids=["unconfirmed", "confirmed"])
def test_a_real_timeout_exception_keeps_the_writer_uncertainty_through_effect_and_preparation(
        tmp_path, monkeypatch, phase, settled):
    work, calls, launches, reaped = _file_work_transport(tmp_path, monkeypatch, phase=phase, settled=settled)
    record, execution = tmp_path / "owner" / "scratch.json", tmp_path / "execution"
    if settled:
        error = subprocess.TimeoutExpired
    else:
        error = work_domain.WorkDomainUnsettled
    with pytest.raises(error) as raised:
        scratch.prepare_session_scratch(record, execution)
    assert len([launch for launch in launches if launch.proc.timed_out]) == 1
    if not settled:
        observed = raised.value.result  # the CompletedProcess every writer of this error carries (query_code reads it)
        assert isinstance(observed, subprocess.CompletedProcess) and observed.returncode is None
        assert observed.backend_settled is False and observed.timed_out is True
        assert observed.stdout == b"partial writer" and observed.stderr == b"deadline"
        assert isinstance(raised.value.__cause__, subprocess.TimeoutExpired)
        assert "without the container confirming" in str(raised.value)
        assert not any(words[0] in {"rm", "rmdir"} for words in calls), "unknown writers are never rolled back"
        assert len(list((work / "scratch").iterdir())) == 1, "an unconfirmed allocation is never retried"
    elif phase != "container":
        assert not list((work / "scratch").iterdir()), "confirmed pre-grant failure may undo its own new names"
    assert not record.exists() and len(reaped) == len(launches)
    assert all(launch.settlements == [settled if launch.proc.timed_out else True] for launch in launches)


@posix_host
def test_new_managed_nonce_is_confined_readable_with_restrictive_umask_and_no_privilege_fallback(tmp_path, monkeypatch):
    from ouroboros import confined_files

    _work, _calls, _launches, _reaped = _file_work_transport(tmp_path, monkeypatch)
    checks = []
    # Enforce the permissions another non-owner UID would need. This is an in-process contract
    # double, not native multi-UID proof; the parent runs that fixture in its isolated Linux kit.
    original_open = confined_files.ConfinedDirectory.open

    def other_uid(self, name):
        stream, info = original_open(self, name)
        if name == scratch.SCRATCH_MARKER:
            checks.append(stat.S_IMODE(info.st_mode))
            if not info.st_mode & stat.S_IROTH:
                stream.close()
                raise PermissionError("synthetic non-owner UID cannot read marker")
        return stream, info

    def forbidden(*_args, **_kwargs):
        raise AssertionError("marker identity never uses cat or a permission/link fallback")

    monkeypatch.setattr(confined_files.ConfinedDirectory, "open", other_uid)
    monkeypatch.setattr(work_domain, "read_in_work", forbidden)
    monkeypatch.setattr(work_domain, "read_bytes", forbidden)
    previous = os.umask(0o077)
    try:
        record = tmp_path / "owner" / "scratch.json"
        workspace = scratch.prepare_session_scratch(record, tmp_path / "execution")
    finally:
        os.umask(previous)
    container = workspace.parent
    assert stat.S_IMODE(container.stat().st_mode) == stat.S_IMODE(container.parent.stat().st_mode) == 0o755
    assert scratch.recorded_scratch(record) == container and checks == [0o644, 0o644]
    assert workspace == container / "w" and list(workspace.iterdir()) == []


def test_a_legacy_identity_survives_only_the_windows_serial_widening():
    """CPython 3.12+ on Windows reports the volume's 64-bit serial where 3.10 reported its low 32 bits: a schema-1
    record made on either side of the move matches there exactly what the 32-bit serial could tell apart."""
    seen = SimpleNamespace(st_dev=(0xABCD << 32) | 0x12345678, st_ino=77)
    assert scratch._legacy_identity_matches(seen, [0x12345678, 77], windows=True)
    assert scratch._legacy_identity_matches(seen, [seen.st_dev, 77], windows=True)
    assert not scratch._legacy_identity_matches(seen, [0x12345679, 77], windows=True)
    assert not scratch._legacy_identity_matches(seen, [0x12345678, 78], windows=True)
    assert not scratch._legacy_identity_matches(seen, [0x12345678, 77], windows=False)
    # The other way: the old checkout ran on 3.12 before its first Update, then the 3.10 app came back.
    narrow = SimpleNamespace(st_dev=0x12345678, st_ino=77)
    assert scratch._legacy_identity_matches(narrow, [seen.st_dev, 77], windows=True)
    assert not scratch._legacy_identity_matches(narrow, [(0xABCD << 32) | 0x12345679, 77], windows=True)
    assert not scratch._legacy_identity_matches(narrow, [seen.st_dev, 77], windows=False)
