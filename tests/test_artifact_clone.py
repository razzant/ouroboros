"""``artifacts.copy_artifact_file`` clones where the filesystem shares blocks (APFS here), streams a copy
elsewhere, never hard-links, and keeps its identity verification either way."""
from __future__ import annotations

import getpass
import hashlib
import json
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ouroboros import artifact_storage, artifacts


def _measured(path):
    raw = path.read_bytes()
    return {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def test_copy_clones_on_this_filesystem_and_the_copy_is_its_own_inode(tmp_path, monkeypatch):
    source, destination = tmp_path / "result.bin", tmp_path / "artifacts" / "result.bin"
    source.write_bytes(os.urandom(1 << 20) * 24)  # 24 MiB of incompressible bytes
    seen, original = [], artifacts.clone_file

    def observed(src, dst):
        outcome = original(src, dst)
        seen.append(outcome)
        return outcome

    monkeypatch.setattr(artifacts, "clone_file", observed)  # the name the copy seam binds
    free_before = shutil.disk_usage(tmp_path).free
    measured = artifacts.copy_artifact_file(source, destination, expected=_measured(source))
    free_after = shutil.disk_usage(tmp_path).free
    print(json.dumps({"case": "clone", "cloned": seen, "free_delta_bytes": free_before - free_after,
                      "size": measured["size"]}), flush=True)
    assert measured == _measured(source) == _measured(destination)
    if sys.platform == "darwin":  # the supported macOS data volume is APFS: the clone path is taken
        assert seen == [True]
    else:
        assert seen in ([True], [False])  # Btrfs/XFS-reflink clone, or the streamed copy elsewhere
    src, dst = os.stat(source), os.stat(destination)
    assert (src.st_dev, src.st_ino) != (dst.st_dev, dst.st_ino) and dst.st_nlink == 1  # never a hard link


def test_a_later_write_to_the_source_never_changes_the_copy(tmp_path):
    source, destination = tmp_path / "result.txt", tmp_path / "out" / "result.txt"
    source.write_bytes(b"first bytes\n" * 4096)
    measured = artifacts.copy_artifact_file(source, destination)
    with source.open("r+b") as handle:  # an in-place rewrite of the source's own blocks
        handle.seek(0)
        handle.write(b"CHANGED")
    with source.open("ab") as handle:
        handle.write(b"appended\n")
    assert _measured(destination) == measured
    assert destination.read_bytes() == b"first bytes\n" * 4096


@pytest.mark.parametrize("refusal", ["false", "oserror"])
def test_an_unavailable_clone_falls_back_to_the_streamed_copy_with_the_same_identity(tmp_path, monkeypatch, refusal):
    source, destination = tmp_path / "result.bin", tmp_path / "out" / "result.bin"
    source.write_bytes(os.urandom(300_000))
    streamed, original = [], artifacts.stream_artifact_file

    def observed(path, sink=None, **kwargs):
        streamed.append(sink is not None)
        return original(path, sink, **kwargs)

    def refused(src, dst):
        if refusal == "oserror":
            raise OSError("clone unavailable")
        return False

    monkeypatch.setattr(artifact_storage, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(artifact_storage, "_clonefile", refused)
    monkeypatch.setattr(artifacts, "stream_artifact_file", observed)
    measured = artifacts.copy_artifact_file(source, destination, expected=_measured(source))
    assert streamed == [True] and measured == _measured(destination) == _measured(source)
    assert list(destination.parent.iterdir()) == [destination]


def test_a_cloned_temporary_of_an_old_source_is_in_flight_for_the_temp_sweep(tmp_path, monkeypatch):
    """Both directions. A clone copies the source's times, so the temporary of a copy of an old result
    looked older than the sweep's age guard while its verification read was still running, and the
    maintenance child's ``temp_sweep`` reaped it from under the copy. The temporary now carries the
    clone's own time: the sweep run during that read keeps it and reaps the genuine orphan beside it,
    the copy completes, and the published file still carries the source's times."""
    from ouroboros.temp_sweep import atomic_temp_path, sweep_stale_temp_files

    root, aged = tmp_path / "data", time.time() - 7200
    source = root / "old_result.bin"
    source.parent.mkdir()
    source.write_bytes(os.urandom(200_000))
    os.utime(source, (aged, aged))
    destination = root / "task_results" / "artifacts" / "t1" / "result.bin"
    destination.parent.mkdir(parents=True)
    orphan = atomic_temp_path(destination.parent / "earlier.bin")  # a hard kill's leftover, genuinely old
    orphan.write_bytes(b"half a copy")
    os.utime(orphan, (aged, aged))
    swept, original = {}, artifacts.stream_artifact_file

    def sweep_while_the_temporary_is_in_flight(path, sink=None, **kwargs):
        temporary = pathlib.Path(sink.name if sink is not None else path)  # streamed into, or cloned then verified
        if temporary.parent == destination.parent and "temporary" not in swept:
            swept["temporary"] = temporary.name
            swept["removed"] = sweep_stale_temp_files(root, scripts=False)
            swept["kept"] = temporary.exists()
        return original(path, sink, **kwargs)

    monkeypatch.setattr(artifacts, "stream_artifact_file", sweep_while_the_temporary_is_in_flight)
    measured = artifacts.copy_artifact_file(source, destination, expected=_measured(source))
    assert swept["temporary"].startswith(".result.bin.tmp.") and (swept["removed"], swept["kept"]) == (1, True)
    assert not orphan.exists() and list(destination.parent.iterdir()) == [destination]
    assert measured == _measured(destination) == _measured(source)
    assert abs(destination.stat().st_mtime - aged) < 2  # the published copy keeps the source's times
    direct = atomic_temp_path(destination.parent / "direct.bin")
    if artifact_storage.clone_file(source, direct):  # the clone seam itself, where the platform copies the times
        assert time.time() - direct.stat().st_mtime < 600


@pytest.mark.parametrize("damaged", ["source", "clone"])
def test_clone_identity_failure_keeps_source_provenance_and_previous_destination(tmp_path, monkeypatch, damaged):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"captured bytes")
    expected = _measured(source)
    destination.write_bytes(b"previous result")
    if damaged == "source":
        source.write_bytes(b"changed bytes")

    def cloned(src, dst):
        dst.write_bytes(b"damaged clone" if damaged == "clone" else src.read_bytes())
        return True

    monkeypatch.setattr(artifacts, "clone_file", cloned)
    with pytest.raises(artifacts.ArtifactIdentityError) as failure:
        artifacts.copy_artifact_file(source, destination, expected=expected)
    if damaged == "source":
        assert failure.value.source_path == source
        stamp = source.stat()
        assert failure.value.source_stamp == (stamp.st_dev, stamp.st_ino, stamp.st_size,
                                               stamp.st_mtime_ns, stamp.st_ctime_ns)
    else:
        assert failure.value.source_path != source  # a bad clone never condemns a valid source
    assert destination.read_bytes() == b"previous result"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["destination", "source"]


def test_identity_verification_is_unchanged_on_the_clone_path(tmp_path):
    source, destination = tmp_path / "result.bin", tmp_path / "out" / "result.bin"
    source.write_bytes(b"x" * 10_000)
    with pytest.raises(artifacts.ArtifactIdentityError, match="sha256"):
        artifacts.copy_artifact_file(source, destination, expected={"size": 10_000, "sha256": "0" * 64})
    assert not destination.exists()
    assert not destination.parent.exists() or not [p for p in destination.parent.iterdir() if p.name.startswith(".")]
    measured = artifacts.copy_artifact_file(source, destination, expected=_measured(source))
    assert measured == _measured(destination)


@pytest.mark.parametrize("kind", ["directory", "fifo"])
def test_a_source_that_is_not_a_regular_file_is_refused_before_any_clone(tmp_path, monkeypatch, kind):
    """A directory would be cloned whole (and its temporary could not be unlinked), a writerless FIFO's
    open would block: neither reaches the clone; the streamed copy's typed refusal answers, nothing is
    left in the destination folder, and a regular file beside them still clones."""
    import threading

    source, destination = tmp_path / "source", tmp_path / "out" / "result"
    if kind == "directory":
        source.mkdir()
        (source / "inner.txt").write_bytes(b"inner")
    else:
        if not hasattr(os, "mkfifo"):
            pytest.skip("no FIFOs on this platform")
        os.mkfifo(source)
    bindings = []
    for name in ("_clonefile", "_ficlone"):
        monkeypatch.setattr(artifact_storage, name, lambda *a, _n=name: bindings.append(_n) or pytest.fail(f"{_n} ran"))
    outcome = {}

    def attempt():
        try:
            artifacts.copy_artifact_file(source, destination)
        except OSError as exc:
            outcome["error"] = str(exc)

    worker = threading.Thread(target=attempt, daemon=True)
    worker.start()
    worker.join(10)
    assert not worker.is_alive(), "the copy blocked on a source that is not a regular file"
    assert "not a regular file" in outcome["error"] and bindings == []
    assert artifact_storage.clone_file(source, tmp_path / "clone") is False and not (tmp_path / "clone").exists()
    assert not destination.exists() and list(destination.parent.iterdir()) == []
    regular = tmp_path / "regular.bin"
    regular.write_bytes(b"bytes" * 1000)
    monkeypatch.undo()
    measured = artifacts.copy_artifact_file(regular, destination, expected=_measured(regular))
    assert measured == _measured(destination) and stat.S_ISREG(destination.stat().st_mode)


def test_the_copy_temporary_carries_the_signature_the_temp_sweep_reaps(tmp_path, monkeypatch):
    source, destination = tmp_path / "source.bin", tmp_path / "out" / "source.bin"
    source.write_bytes(b"x" * 4096)
    names = []
    monkeypatch.setattr(artifacts, "clone_file", lambda src, dst: names.append(dst.name) or False)
    artifacts.copy_artifact_file(source, destination, expected=_measured(source))
    [name] = names
    assert name.startswith(".source.bin.tmp.") and not set(name.rsplit(".tmp.", 1)[1]) - set("0123456789abcdef.")


def test_a_refused_linux_clone_leaves_no_empty_destination(tmp_path, monkeypatch):
    if not sys.platform.startswith("linux"):
        pytest.skip("FICLONE is a Linux ioctl")
    import fcntl

    source, destination = tmp_path / "a", tmp_path / "b"
    source.write_bytes(b"bytes")
    monkeypatch.setattr(fcntl, "ioctl", lambda *a: (_ for _ in ()).throw(OSError(95, "not supported")))
    assert artifact_storage.clone_file(source, destination) is False and not destination.exists()


# ---- clone_or_copy2: the reviewer folders' copytree function ---------------------------------------------------

darwin = pytest.mark.skipif(sys.platform != "darwin", reason="real APFS clones")
_BYTES = b"bytes the copy keeps\n" * 8192
_AGED = (1_600_000_000_123_456_789, 1_600_000_001_987_654_321)


def _aged_source(path):
    path.write_bytes(_BYTES)
    path.chmod(0o640)
    os.utime(path, ns=_AGED)
    return path


def _copy2_facts(path):
    info = os.lstat(path)
    return {"bytes": path.read_bytes(), "mode": info.st_mode, "mtime_ns": info.st_mtime_ns, "nlink": info.st_nlink}


def test_clone_or_copy2_keeps_copy2s_bytes_mode_and_times_and_shares_blocks_where_it_can(tmp_path):
    source = _aged_source(tmp_path / "source")
    shutil.copy2(source, tmp_path / "streamed")
    artifact_storage.clone_or_copy2(str(source), str(tmp_path / "cloned"))  # copytree hands its function strings
    assert _copy2_facts(tmp_path / "cloned") == _copy2_facts(tmp_path / "streamed") == _copy2_facts(source)
    assert os.stat(source).st_ino != os.stat(tmp_path / "cloned").st_ino
    if sys.platform == "darwin":
        assert artifact_storage._may_share(tmp_path / "cloned") is True
        assert artifact_storage._may_share(tmp_path / "streamed") is False


@darwin
@pytest.mark.parametrize("side", ["source", "clone"])
def test_a_write_to_either_side_of_a_real_clone_leaves_the_other_whole(tmp_path, side):
    source = _aged_source(tmp_path / "source")
    clone = tmp_path / "clone"
    artifact_storage.clone_or_copy2(source, clone)
    assert artifact_storage._may_share(clone) is True, "this proves a clone, not a copy"
    written, other = (source, clone) if side == "source" else (clone, source)
    with written.open("r+b") as handle:  # in place, through the shared blocks' own inode
        handle.seek(4096)
        handle.write(b"CHANGED")
    with written.open("ab") as handle:
        handle.write(b"appended\n")
    assert other.read_bytes() == _BYTES and os.lstat(other).st_mtime_ns == _AGED[1]
    assert written.read_bytes() != _BYTES and written.read_bytes().endswith(b"appended\n")


@pytest.mark.parametrize("refusal", ["false", "oserror"])
def test_a_refused_clone_is_copy2_with_nothing_left_behind(tmp_path, monkeypatch, refusal):
    source = _aged_source(tmp_path / "source")

    def refused(src, dst):
        if refusal == "oserror":
            raise OSError("clone unavailable")
        return False

    monkeypatch.setattr(artifact_storage, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(artifact_storage, "_clonefile", refused)
    artifact_storage.clone_or_copy2(source, tmp_path / "copied")
    assert _copy2_facts(tmp_path / "copied") == _copy2_facts(source)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["copied", "source"]


def test_an_existing_destination_is_overwritten_as_copy2_does(tmp_path):
    """``dirs_exist_ok`` merges a frozen view into a Git copy: the clone refuses an existing name, copy2 answers."""
    source = _aged_source(tmp_path / "source")
    (tmp_path / "there").write_bytes(b"older")
    artifact_storage.clone_or_copy2(source, tmp_path / "there")
    assert _copy2_facts(tmp_path / "there") == _copy2_facts(source)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs on this platform")
def test_a_special_file_gets_copy2s_typed_refusal(tmp_path):
    os.mkfifo(tmp_path / "pipe")
    with pytest.raises(shutil.SpecialFileError):
        artifact_storage.clone_or_copy2(tmp_path / "pipe", tmp_path / "copied")
    assert not (tmp_path / "copied").exists()


@darwin
def test_an_immutable_source_is_copied_because_its_clone_would_refuse_the_source_times(tmp_path):
    """``clonefile`` carries ``uchg``, and an immutable clone takes no times: copy2 sets them in an order it can."""
    source = _aged_source(tmp_path / "source")
    os.chflags(source, stat.UF_IMMUTABLE)
    copied = tmp_path / "copied"
    try:
        artifact_storage.clone_or_copy2(source, copied)
        assert _copy2_facts(copied) == _copy2_facts(source)
        assert artifact_storage._may_share(copied) is False
    finally:
        os.chflags(source, 0)
        if copied.exists():
            os.chflags(copied, 0)


@darwin
def test_a_compressed_source_stays_readable_and_compressed_in_its_clone(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"the same line again and again\n" * 16384)
    assert artifact_storage.compression_outcome(source) == "compressed"
    original = source.read_bytes()
    clone = tmp_path / "clone"
    artifact_storage.clone_or_copy2(source, clone)
    assert clone.read_bytes() == original and os.lstat(clone).st_flags & stat.UF_COMPRESSED
    with clone.open("r+b") as handle:  # a seat's edit in place: the kernel decompresses its own copy first
        handle.seek(100)
        handle.write(b"EDIT")
    assert clone.read_bytes() == original[:100] + b"EDIT" + original[104:] and source.read_bytes() == original


@pytest.fixture
def deny():
    """Sets a deny ACE and takes it off again: a source nobody may delete would outlive its test's scratch."""
    denied = []

    def apply(path, *rights):
        denied.append(path)
        subprocess.run(["chmod", "+a", f"{getpass.getuser()} deny {','.join(rights)}", str(path)], check=True)

    yield apply
    for path in denied:
        subprocess.run(["chmod", "-N", str(path)], check=True)


def _acl(path):
    return subprocess.run(["ls", "-led", str(path)], check=True, capture_output=True, text=True).stdout.splitlines()[1:]


@darwin
@pytest.mark.parametrize("rights", [("write", "append"), ("writeattr",), ("writeextattr",), ("writesecurity",), ("delete",)])
def test_a_denying_source_acl_reaches_no_copy_and_a_seat_still_edits_its_own(tmp_path, deny, rights):
    """The source stays a real clone source with its mode writable, and its ACL stays behind as ``copy2`` leaves it:
    the times are set, the seat writes, and the copy goes with its folder."""
    source = _aged_source(tmp_path / "source")
    deny(source, *rights)
    assert _acl(source), "the fixture really denies"
    (tmp_path / "seat").mkdir()
    copies = {"cloned": artifact_storage.clone_or_copy2, "streamed": shutil.copy2}
    for name, copier in copies.items():
        copier(source, tmp_path / "seat" / name)
    assert artifact_storage._may_share(tmp_path / "seat" / "cloned") is True, "this proves a clone, not a copy"
    for name in copies:
        copy = tmp_path / "seat" / name
        assert _acl(copy) == [] and os.lstat(copy).st_mtime_ns == _AGED[1] and stat.S_IMODE(os.lstat(copy).st_mode) == 0o640
        with copy.open("r+b") as handle:
            handle.write(b"EDIT")
        os.utime(copy)
        os.chmod(copy, 0o600)
        assert copy.read_bytes().startswith(b"EDIT")
    shutil.rmtree(tmp_path / "seat")


@darwin
def test_a_source_acl_that_refuses_the_clone_is_copy2(tmp_path, deny):
    """``clonefile`` must read the attributes it copies: a refusal this way is the ordinary fallback."""
    source = _aged_source(tmp_path / "source")
    deny(source, "readextattr")
    artifact_storage.clone_or_copy2(source, tmp_path / "copied")
    assert artifact_storage._may_share(tmp_path / "copied") is False
    assert _copy2_facts(tmp_path / "copied") == _copy2_facts(source)


@darwin
def test_a_destination_acl_is_inherited_by_the_clone_as_by_copy2(tmp_path, deny):
    source = _aged_source(tmp_path / "source")
    folder = tmp_path / "folder"
    folder.mkdir()
    deny(folder, "writeextattr", "file_inherit")
    artifact_storage.clone_or_copy2(source, folder / "cloned")
    shutil.copy2(source, folder / "streamed")
    assert artifact_storage._may_share(folder / "cloned") is True
    assert _acl(folder / "cloned") == _acl(folder / "streamed") != []
