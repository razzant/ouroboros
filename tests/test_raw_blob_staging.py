"""Raw-byte staging through ONE ``git fast-import`` (``stage_raw_payload_inventory``).

The oracle for every staged entry is independent of the writer: the Git blob id
recomputed from the file's raw bytes (or a symlink's raw target) and the stored
object's bytes read back. The three real consumers (reviewer capture, skill
payload provisioning and capture) are driven end to end at a scale where a Git
process per file would be visible in the process count. Git's refusal must
keep capture copies, or discard a never-started snapshot while preserving its original.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shlex
import stat
import subprocess
import sys

import pytest

from ouroboros import review_checking as rc
from ouroboros.subagent_worktrees import RawStagingIncomplete, isolated_git_env, stage_raw_payload_inventory
from ouroboros.unindexed import COPIES
from tests.group_run import run_in_group
from tests.test_delegated_skill_payload import (  # noqa: F401
    _owned_gateway_uses_each_test_transport,
    _payload_ctx,
    _payload_entry,
    _seed_skill,
)
from tests.test_review_checking import _git, author  # noqa: F401

POSIX = os.name != "nt"
REPO = pathlib.Path(__file__).resolve().parents[1]
# A loader-visible file on every platform whose path every Git refuses to index (a .git component).
_REFUSED = "vendored/.Git/info.txt"


def _blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _repo(tmp_path: pathlib.Path):
    root = tmp_path / "repo"
    root.mkdir()
    env = isolated_git_env()
    subprocess.run(["git", "init", "-q", "--template="], cwd=root, env=env, check=True, capture_output=True)
    return root, env


def _staged(root, env, *, ref=":0") -> dict:
    listed = subprocess.run(["git", "ls-files", "--stage", "-z"] if ref == ":0" else
                            ["git", "ls-tree", "-r", "-z", ref],
                            cwd=root, env=env, check=True, capture_output=True).stdout
    entries = {}
    for chunk in listed.split(b"\0"):
        if chunk:
            meta, _, name = chunk.partition(b"\t")
            fields = meta.split()
            entries[os.fsdecode(name)] = (fields[0].decode(), fields[1 if ref == ":0" else 2].decode())
    return entries


def _raw(path: pathlib.Path) -> bytes:
    return os.readlink(os.fsencode(str(path))) if path.is_symlink() else path.read_bytes()


def _verb(args: list) -> str:
    """A git command's subcommand, past any leading ``-c key=value`` pairs."""
    rest = args[1:]
    while rest[:1] == ["-c"]:
        rest = rest[2:]
    return rest[0] if args[:1] == ["git"] and rest else ""


class _Recorder:
    """Records every process this test starts; optionally swaps the blob writer."""

    def __init__(self, monkeypatch, fake_writer: str = ""):
        self.args, self.procs, real = [], [], subprocess.Popen
        recorder = self

        class Recording(real):
            def __init__(self, args, *rest, **kwargs):
                args = list(args)
                recorder.args.append(args)
                if fake_writer and _verb(args) == "fast-import":
                    args = [sys.executable, "-c", fake_writer]
                super().__init__(args, *rest, **kwargs)
                recorder.procs.append(self)

        monkeypatch.setattr(subprocess, "Popen", Recording)

    def git(self, verb: str) -> int:
        return sum(1 for args in self.args if _verb(args) == verb or args[:1] == ["git"] and verb in args[1:6])


_FRAMING = [b"blob\nmark :1\ndata 3\nabc\nget-mark :1\ndone\n", b"done\n", b"\n", b"data 99999\n"]


def test_raw_staging_ignores_hostile_attributes_filters_and_config(tmp_path):
    root, env = _repo(tmp_path)
    marker = tmp_path / "filter-ran"
    probe = f"sh -c 'echo ran >> {shlex.quote(str(marker))}; cat'"
    for key, value in (("filter.probe.clean", probe), ("filter.probe.smudge", probe),
                       ("filter.probe.required", "true"), ("core.autocrlf", "true"),
                       ("core.eol", "crlf"), ("core.safecrlf", "true"),
                       ("core.bigFileThreshold", "1")):  # writer overrides this; big.bin covers streaming
        subprocess.run(["git", "config", key, value], cwd=root, env=env, check=True)
    files = {
        ".gitattributes": b"* text eol=crlf filter=probe diff=probe\n*.u16 working-tree-encoding=UTF-16\n",
        # Not nul.bin: newer Windows creates that file, but Git for Windows refuses a device name as an index
        # path even with an extension (test_a_windows_device_name_is_staged_or_refused_never_dropped).
        "crlf.txt": b"one\r\ntwo\r\n", "lf.txt": b"one\ntwo", "empty": b"", "binary.bin": b"\x00\xff\xfe\r\n",
        "text.u16": "héllo\r\n".encode("utf-8"), "big.bin": bytes(range(256)) * 12_289,
        "-dash name.txt": b"leading dash", "ü/naïve.txt": b"precomposed",
        **{f"framing/{i}": payload for i, payload in enumerate(_FRAMING)},
    }
    if POSIX:
        files["odd\nname.txt"] = b"newline in its name\r\n"
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    if POSIX:
        os.chmod(root / "lf.txt", 0o755)
        (root / "dir").mkdir()
        for name, target in (("rel-link", "crlf.txt"), ("abs-link", str(root / "lf.txt")),
                             ("dangling", "missing\nname"), ("dir-link", "dir")):
            os.symlink(target, root / name)
    rels = [p.relative_to(root).as_posix() for p in root.rglob("*")
            if ".git" not in p.relative_to(root).parts and (p.is_symlink() or p.is_file())]

    assert stage_raw_payload_inventory(root, rels, env) == []

    staged = _staged(root, env)
    assert set(staged) == set(rels)
    for rel, (mode, oid) in staged.items():
        path, raw = root / rel, _raw(root / rel)
        expected_mode = "120000" if path.is_symlink() else (
            "100755" if path.stat().st_mode & 0o111 else "100644")
        assert (mode, oid) == (expected_mode, _blob_id(raw)), rel
        shown = subprocess.run(["git", "cat-file", "blob", oid], cwd=root, env=env,
                               check=True, capture_output=True).stdout
        assert shown == raw, rel
    assert not marker.exists()  # no clean/smudge/process filter ran



def test_a_payload_capture_leaves_no_control_folder_where_read_only_files_refuse_deletion(tmp_path, monkeypatch):
    """Windows refuses to delete a read-only file, and the blobs a capture stages into its
    host-managed control object database are read-only: the control folder still goes instead
    of staying in the temp folder forever."""
    from ouroboros.subagent_worktrees import payload_capture_git_env

    root, _env = _repo(tmp_path)
    (root / "new.txt").write_bytes(b"staged into the control object database")
    real_unlink = os.unlink

    def windows_unlink(path, *, dir_fd=None):
        if not os.lstat(path, dir_fd=dir_fd).st_mode & stat.S_IWRITE:
            raise PermissionError(13, "Access is denied", path)
        return real_unlink(path, dir_fd=dir_fd)

    with payload_capture_git_env(root) as env:
        control = pathlib.Path(env["GIT_DIR"]).parent
        subprocess.run(["git", "hash-object", "-w", "new.txt"], cwd=root, env=env, check=True, capture_output=True)
        objects = [p for p in (control / ".git" / "objects").rglob("*") if p.is_file()]
        assert any(not p.stat().st_mode & stat.S_IWRITE for p in objects), "git staged read-only objects"
        monkeypatch.setattr(os, "unlink", windows_unlink)
        monkeypatch.setattr(os, "remove", windows_unlink)
    assert not control.exists()

@pytest.mark.skipif(not POSIX, reason="Windows has no POSIX executable bit to diverge")
def test_baseline_modes_pin_regular_files_and_report_divergence(tmp_path):
    root, env = _repo(tmp_path)
    for name, mode in (("kept-exec", 0o755), ("lost-exec", 0o644), ("gained-exec", 0o755),
                       ("new-exec", 0o755), ("was-link", 0o644)):
        (root / name).write_bytes(name.encode())
        os.chmod(root / name, mode)
    baseline = {"kept-exec": "100755", "lost-exec": "100755", "gained-exec": "100644",
                "was-link": "120000"}

    divergent = stage_raw_payload_inventory(root, list(baseline) + ["new-exec"], env,
                                            baseline_modes=baseline)

    assert divergent == ["gained-exec", "lost-exec", "new-exec"]
    assert {rel: mode for rel, (mode, _oid) in _staged(root, env).items()} == {
        "kept-exec": "100755", "lost-exec": "100755", "gained-exec": "100644",
        "new-exec": "100644", "was-link": "100644"}


# The staging under measurement runs alone in a fresh interpreter: tracemalloc traces every thread
# of its process, and an xdist worker still runs other tests' background threads, which allocate
# inside the window and were counted as this call's memory (a red on a loaded CI runner).
_MEASURED_STAGING = r"""
import json, pathlib, subprocess, sys, tracemalloc
from ouroboros.subagent_worktrees import isolated_git_env, stage_raw_payload_inventory

root, rels = pathlib.Path(sys.argv[1]), json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
env, started, real = isolated_git_env(), [], subprocess.Popen

class Recording(real):
    def __init__(self, args, *rest, **kwargs):
        started.append([str(arg) for arg in args])
        super().__init__(args, *rest, **kwargs)

subprocess.Popen = Recording
tracemalloc.start()
try:
    divergent = stage_raw_payload_inventory(root, rels, env)
    peak = tracemalloc.get_traced_memory()[1]
finally:
    tracemalloc.stop()
print(json.dumps({"divergent": divergent, "peak": peak, "started": started}))
"""


def test_thousands_of_files_use_one_writer_process_and_never_hold_the_tree(tmp_path):
    root, env = _repo(tmp_path)
    count, size = 3000, 16 * 1024
    for i in range(count):
        path = root / f"d{i % 40:02d}" / f"f{i:05d}.bin"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(i.to_bytes(4, "big") * (size // 4))
    rels = [p.relative_to(root).as_posix() for p in root.rglob("*.bin")]
    listing = tmp_path / "rels.json"
    listing.write_text(json.dumps(rels), encoding="utf-8")
    done = run_in_group([sys.executable, "-c", _MEASURED_STAGING, str(root), str(listing)],
                        timeout=240, cwd=REPO)
    assert done.returncode == 0, done.stdout + done.stderr
    measured = json.loads(done.stdout.strip().splitlines()[-1])
    assert measured["divergent"] == []

    # Constant, whatever the count: one writer, one staging call, one index read proving every record landed.
    assert [_verb(args) for args in measured["started"]] == ["fast-import", "update-index", "ls-files"]
    # A whole-tree buffer would hold count*size (48 MiB) at once; one file at a time stays far below.
    assert measured["peak"] < count * size // 8, measured["peak"]
    staged = _staged(root, env)
    assert len(staged) == count
    assert staged["d07/f00007.bin"] == ("100644", _blob_id((7).to_bytes(4, "big") * (size // 4)))


def test_large_blobs_stream_whole_into_the_one_kept_pack(tmp_path):
    root, env = _repo(tmp_path)
    big = hashlib.sha256(b"seed").digest() * (2 * 65536)  # 4 MiB, over the 1 MiB stream threshold
    files = {"a.txt": b"small\r\n", "big1.bin": big, "big2.bin": big[:-1] + b"!"}
    for rel, data in files.items():
        (root / rel).write_bytes(data)

    stage_raw_payload_inventory(root, list(files), env)

    objects = root / ".git" / "objects"
    packs = list((objects / "pack").glob("*.pack"))
    assert len(packs) == 1 and not list(objects.glob("??/*"))  # kept as written, never exploded
    verified = subprocess.run(["git", "verify-pack", "-v", str(packs[0])], cwd=root, env=env,
                              check=True, capture_output=True, text=True).stdout
    rows = {line.split()[0]: line.split() for line in verified.splitlines() if " blob " in line}
    # A blob held in memory is deltified against its predecessor; a streamed one never is.
    assert len(rows[_blob_id(files["big2.bin"])]) == 5, rows
    assert {oid for _mode, oid in _staged(root, env).values()} == {_blob_id(d) for d in files.values()}


def _seeded_index(tmp_path):
    root, env = _repo(tmp_path)
    for name in ("a.txt", "b.txt", "c.txt"):
        (root / name).write_bytes(name.encode() * 3)
    stage_raw_payload_inventory(root, ["a.txt"], env)
    return root, env, (root / ".git" / "index").read_bytes()


def test_a_read_failure_mid_stream_kills_the_writer_and_stages_nothing(tmp_path, monkeypatch):
    root, env, index = _seeded_index(tmp_path)
    (root / "b.txt").unlink()  # vanished between the caller's inventory and staging
    recorder = _Recorder(monkeypatch)
    with pytest.raises(FileNotFoundError):
        stage_raw_payload_inventory(root, ["a.txt", "b.txt", "c.txt"], env)
    assert (root / ".git" / "index").read_bytes() == index
    assert recorder.git("update-index") == 0
    assert [proc.returncode is not None for proc in recorder.procs] == [True]  # reaped, not orphaned


def test_a_killed_writer_fails_typed_and_stages_nothing(tmp_path, monkeypatch):
    root, env, index = _seeded_index(tmp_path)
    recorder, real_read = _Recorder(monkeypatch), pathlib.Path.read_bytes

    def read_then_kill(path):
        if path.name == "b.txt":
            recorder.procs[0].kill()
            recorder.procs[0].wait()
        return real_read(path)

    monkeypatch.setattr(pathlib.Path, "read_bytes", read_then_kill)
    with pytest.raises(subprocess.CalledProcessError) as raised:
        stage_raw_payload_inventory(root, ["a.txt", "b.txt", "c.txt"], env)
    assert raised.value.returncode != 0
    assert (root / ".git" / "index").read_bytes() == index
    assert recorder.git("update-index") == 0


# Stand-ins for a writer that misbehaves after reading a correct stream: valid-looking
# ids must never be staged unless the stream reached ``done`` and exited 0.
_FAKE_PARSER = """
import hashlib, sys
inp, out = sys.stdin.buffer, sys.stdout.buffer
while True:
    line = inp.readline()
    if line != b"blob\\n":
        break
    inp.readline()
    size = int(inp.readline().split()[1])
    data = inp.read(size)
    inp.readline(); inp.readline()
    out.write(ANSWER(size, data) + b"\\n"); out.flush()
"""
_IDS = "ANSWER = lambda size, data: hashlib.sha1(b'blob %d\\0' % size + data).hexdigest().encode()\n"


@pytest.mark.parametrize("fake", [
    "import sys; sys.exit(0)",  # exits before answering
    "ANSWER = lambda size, data: b'not-an-object-id'\n" + _FAKE_PARSER,
    _IDS + _FAKE_PARSER + "sys.exit(1)\n",  # every id answered, then the stream fails at done
    _IDS + _FAKE_PARSER + "out.write(b'unexpected'); out.flush()\n",
], ids=["silent-exit", "bad-id", "fails-at-done", "trailing-output"])
def test_a_misbehaving_writer_is_never_staged(tmp_path, monkeypatch, fake):
    root, env, index = _seeded_index(tmp_path)
    recorder = _Recorder(monkeypatch, fake_writer=fake)
    with pytest.raises(subprocess.CalledProcessError):
        stage_raw_payload_inventory(root, ["a.txt", "b.txt", "c.txt"], env)
    assert (root / ".git" / "index").read_bytes() == index
    assert recorder.git("update-index") == 0
    assert all(proc.returncode is not None for proc in recorder.procs)


def test_records_git_refuses_or_replaces_are_never_a_normal_return(tmp_path):
    root, env = _repo(tmp_path)
    for rel, data in (("keep.txt", b"old\n"), ("sub/.keep", b""), (_REFUSED, b"inside\n"), ("added.txt", b"added\n")):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    assert stage_raw_payload_inventory(root, ["keep.txt"], env) == []
    (root / "keep.txt").write_bytes(b"new\n")
    # Real ``update-index --index-info`` exits 0 while skipping each path it refuses: the original helper returned [].
    with pytest.raises(RawStagingIncomplete) as raised:
        stage_raw_payload_inventory(root, ["added.txt", "sub/../keep.txt", _REFUSED], env)
    assert isinstance(raised.value, subprocess.CalledProcessError) and raised.value.returncode == 0
    assert f"lacks 2 of 3 records as sent (sub/../keep.txt, {_REFUSED})" in str(raised.value), raised.value
    staged = _staged(root, env)
    assert staged["keep.txt"] == ("100644", _blob_id(b"old\n"))  # the new bytes never reached the index
    assert staged["added.txt"] == ("100644", _blob_id(b"added\n")) and _REFUSED not in staged
    if POSIX:  # a later record replacing an earlier one (file/directory) is no refusal at all; the records show it
        os.symlink("sub", root / "link")
        with pytest.raises(RawStagingIncomplete, match=r"lacks 1 of 2 records as sent \(link\)"):
            stage_raw_payload_inventory(root, ["link", "link/.keep"], env)


def test_a_windows_device_name_is_staged_or_refused_never_dropped(tmp_path):
    root, env = _repo(tmp_path)
    data = b"\x00\xff\xfe\r\n"
    (root / "nul.bin").write_bytes(data)
    if "nul.bin" not in os.listdir(root):
        assert os.name == "nt"  # only older Windows hands the write to the NUL device
        pytest.skip("this Windows maps nul.bin to the NUL device: no file exists to stage")
    try:
        divergent = stage_raw_payload_inventory(root, ["nul.bin"], env)
    except RawStagingIncomplete as refused:  # Git for Windows refuses a device name as an index path
        assert os.name == "nt" and "nul.bin" in str(refused), refused
        assert "nul.bin" not in _staged(root, env)
    else:
        assert divergent == [] and _staged(root, env) == {"nul.bin": ("100644", _blob_id(data))}


def _bulk(root: pathlib.Path, count: int, prefix: str) -> dict:
    files = {}
    for i in range(count):
        rel = f"{prefix}/{i % 25:02d}/n{i:04d}.txt"
        data = (b"row %d\r\n" % i) * 3 + (b"nul\x00ok\n" if i % 4 == 0 else b"")
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
        files[rel] = data
    return files


def test_reviewer_capture_of_a_large_tree_keeps_raw_bytes_with_one_writer(author, tmp_path, monkeypatch):  # noqa: F811
    data = tmp_path / "data"
    source = rc.freeze_checking_source(author, data_root=data, subject=rc.SUBJECT_INDEX)
    area = pathlib.Path(rc.prepare_seat_area(data, "task", "bulk", source))
    files = _bulk(area, 1500, "bulk")
    (area / "f.txt").write_bytes(b"edited\r\n\xff")
    files["f.txt"] = b"edited\r\n\xff"
    recorder = _Recorder(monkeypatch)

    evidence = rc.capture_seat_evidence(data, "task", "bulk")

    assert evidence["status"] == "retained", evidence
    assert recorder.git("fast-import") == 1 and recorder.git("hash-object") == 0
    assert len(recorder.args) < 30  # a process per file would be > 1500
    replay = tmp_path / "replay"
    rc._materialize_git(replay, source)
    _git(replay, "apply", "--cached", "--binary", str(pathlib.Path(evidence["evidence_dir"]) / "delta.patch"))
    staged = _staged(replay, isolated_git_env())
    assert {rel: staged[rel][1] for rel in files} == {rel: _blob_id(raw) for rel, raw in files.items()}


def test_payload_provision_and_capture_stage_raw_bytes_with_one_writer_each(tmp_path, monkeypatch):
    from ouroboros import delegate_custody as custody
    from ouroboros.subagent_worktrees import provision_payload_snapshot
    from ouroboros.tools.delegate import _capture_terminal_patch
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx = _payload_ctx(tmp_path, monkeypatch)
    skill = _seed_skill(tmp_path / "data")
    original = _bulk(skill, 600, "res")
    recorder = _Recorder(monkeypatch)
    handle = provision_payload_snapshot(target_root=skill, task_id="t-payload", snapshot_id="snapP")
    assert recorder.git("fast-import") == 1 and recorder.git("hash-object") == 0
    baseline = _staged(pathlib.Path(handle.path), isolated_git_env(), ref=handle.baseline_tree)
    assert {rel: baseline[rel][1] for rel in original} == {rel: _blob_id(raw) for rel, raw in original.items()}

    exec_root = pathlib.Path(handle.path)
    edited = {rel: raw.replace(b"row", b"ROW") for rel, raw in list(original.items())[:300]}
    for rel, raw in edited.items():
        (exec_root / rel).write_bytes(raw)
    edited.update(_bulk(exec_root, 200, "added"))
    custody._CUSTODY.clear()
    entry = _payload_entry(handle, skill)
    recorder.args.clear()
    capture = _capture_terminal_patch(ctx, entry)

    assert capture["status"] == "ready_with_changes", capture
    assert recorder.git("fast-import") == 1 and recorder.git("hash-object") == 0
    assert len(recorder.args) < 30  # a process per file would be > 800
    manifest = json.loads(pathlib.Path(capture["manifest_artifact"]).read_text(encoding="utf-8"))
    assert set(manifest["tracked_changed"]) == set(edited)
    out = _integrate_delegated_patch(ctx, "run-p1", "apply", "")
    assert "✅ Integrated" in out, out
    assert all((skill / rel).read_bytes() == raw for rel, raw in edited.items())
    custody._CUSTODY.clear()


def test_payload_capture_read_failure_is_a_typed_failed_manifest(tmp_path, monkeypatch):
    from ouroboros import delegate_custody as custody
    from ouroboros.subagent_worktrees import provision_payload_snapshot
    from ouroboros.tools.delegate import _capture_terminal_patch

    ctx = _payload_ctx(tmp_path, monkeypatch)
    skill = _seed_skill(tmp_path / "data")
    handle = provision_payload_snapshot(target_root=skill, task_id="t-payload", snapshot_id="snapP")
    exec_root = pathlib.Path(handle.path)
    (exec_root / "notes.txt").write_bytes(b"DONE\r\n")
    real_read = pathlib.Path.read_bytes

    def vanish_then_read(path):
        if path == exec_root / "plugin.py":  # listed by the loader, gone when staging reads it
            raise FileNotFoundError(2, "No such file or directory", str(path))
        return real_read(path)

    monkeypatch.setattr(pathlib.Path, "read_bytes", vanish_then_read)
    custody._CUSTODY.clear()
    capture = _capture_terminal_patch(ctx, _payload_entry(handle, skill))

    assert capture["status"] == "failed", capture
    note = json.loads(pathlib.Path(capture["manifest_artifact"]).read_text(encoding="utf-8"))["note"]
    assert note.startswith("staging failed: FileNotFoundError"), note
    assert not (pathlib.Path(capture["manifest_artifact"]).parent / "workspace.patch").exists()
    custody._CUSTODY.clear()


def test_reviewer_capture_of_a_path_git_refuses_keeps_the_copy(author, tmp_path, monkeypatch):  # noqa: F811
    data = tmp_path / "data"
    source = rc.freeze_checking_source(author, data_root=data, subject=rc.SUBJECT_INDEX)
    area = pathlib.Path(rc.prepare_seat_area(data, "task", "refused", source))
    (area / "f.txt").write_bytes(b"reviewed\n")
    (area / _REFUSED).parent.mkdir(parents=True)
    (area / _REFUSED).write_bytes(b"reviewer output\n")
    # Natively a reviewer's nul.bin on Windows. Where core.ignorecase is true, Git's untracked walk skips a .Git
    # directory before staging (a separate inventory gap), so the request is completed here; the helper and
    # Git's refusal stay real.
    monkeypatch.setattr("ouroboros.subagent_worktrees.stage_raw_payload_inventory",
                        lambda root, paths, env: stage_raw_payload_inventory(root, {*paths, _REFUSED}, env))

    evidence = rc.capture_seat_evidence(data, "task", "refused")

    assert evidence["status"] == "capture_failed" and "RawStagingIncomplete" in evidence["error"], evidence
    assert _REFUSED in evidence["error"] and area.is_dir()  # never released over a dropped file
    assert (area / _REFUSED).read_bytes() == b"reviewer output\n" and (area / "f.txt").read_bytes() == b"reviewed\n"


def test_payload_capture_of_a_path_git_refuses_fails_typed_and_keeps_the_snapshot(tmp_path, monkeypatch):
    from ouroboros import delegate_custody as custody
    from ouroboros.subagent_worktrees import find_execution_snapshot, provision_payload_snapshot
    from ouroboros.tools.delegate import _capture_terminal_patch
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx = _payload_ctx(tmp_path, monkeypatch)
    skill = _seed_skill(tmp_path / "data")
    handle = provision_payload_snapshot(target_root=skill, task_id="t-payload", snapshot_id="snapP")
    exec_root = pathlib.Path(handle.path)
    (exec_root / "notes.txt").write_bytes(b"DONE\n")
    (exec_root / _REFUSED).parent.mkdir(parents=True)
    (exec_root / _REFUSED).write_bytes(b"loader-visible\n")
    custody._CUSTODY.clear()
    entry = _payload_entry(handle, skill)
    capture = _capture_terminal_patch(ctx, entry)

    assert capture["status"] == "failed" and entry.patch_captured is False, capture
    manifest = pathlib.Path(capture["manifest_artifact"])
    note = json.loads(manifest.read_text(encoding="utf-8"))["note"]
    assert note.startswith("staging failed: RawStagingIncomplete") and _REFUSED in note, note
    assert not (manifest.parent / "workspace.patch").exists()
    assert "INTEGRATE_DELEGATED_CAPTURE_FAILED" in _integrate_delegated_patch(ctx, "run-p1", "apply", "")
    assert (skill / "notes.txt").read_text(encoding="utf-8") == "PENDING\n" and not (skill / "vendored").exists()
    assert find_execution_snapshot("snapP") is not None and (exec_root / _REFUSED).is_file()  # snapshot kept
    custody._CUSTODY.clear()


def test_payload_provisioning_of_a_path_git_refuses_never_starts_or_leaves_a_snapshot(tmp_path, monkeypatch):
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_shared import delegate_payload
    from ouroboros.subagent_worktrees import find_execution_snapshot
    from ouroboros.tools.delegate_integration import _provision_payload_snapshot

    ctx = _payload_ctx(tmp_path, monkeypatch)
    skill = _seed_skill(tmp_path / "data")
    (skill / _REFUSED).parent.mkdir(parents=True)
    (skill / _REFUSED).write_bytes(b"loader-visible\n")

    record = {"target_root": str(skill), "resource_ref": {}}
    handle, refusal = _provision_payload_snapshot(ctx, custody.custody_root(ctx), record, "snapR")

    assert handle is None, handle  # a ready snapshot would carry a baseline without the refused file
    payload = delegate_payload(refusal)
    assert payload["reason"] == "execution_snapshot_failed" and payload["definitely_unrun"] is True, payload
    assert "RawStagingIncomplete" in payload["detail"] and _REFUSED in payload["detail"], payload
    assert not [*(tmp_path / "snaps").glob("dlgp_*"), *(tmp_path / "snaps").glob(f"{COPIES}/dlgp_*")]
    assert find_execution_snapshot("snapR") is None
    assert not (skill / ".git").exists() and (skill / _REFUSED).read_bytes() == b"loader-visible\n"
